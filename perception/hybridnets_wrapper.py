"""
perception/hybridnets_wrapper.py
=================================
Production HybridNets perception module for NeuroDrive-XAI.

Supports:
  - ONNX Runtime (primary, fastest)
  - Native PyTorch fallback (if ONNX export not available)
  - Automatic ONNX conversion via setup_models.py
  - TensorRT → CUDA → CPU provider cascade
  - Returns feature tensors for downstream GradCAM XAI
"""

import sys
import os
import cv2
import numpy as np
from utils.profiler import PerfTimer

# Ensure HybridNets is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "HybridNets"))

try:
    from utils.utils import letterbox, scale_coords, postprocess, BBoxTransform, ClipBoxes, Params
    HYBRIDNETS_AVAILABLE = True
except ImportError:
    HYBRIDNETS_AVAILABLE = False
    print("[PerceptionModule] WARNING: HybridNets utils not found. Using fallback detector.")


class PerceptionModule:
    """
    Multi-task perception: object detection + lane segmentation + drivable area.
    Primary backbone: HybridNets (EfficientDet + BiFPN) trained on BDD100K.
    """

    def __init__(
        self,
        model_path: str = "weights/hybridnets.onnx",
        project_file: str = "HybridNets/projects/bdd100k.yml",
        use_cuda: bool = True,
    ):
        self.timer = PerfTimer("Perception-Total")
        self.inf_timer = PerfTimer("Perception-Inference")
        self.use_cuda = use_cuda

        # Object & segmentation class lists
        if HYBRIDNETS_AVAILABLE and os.path.exists(project_file):
            self.params = Params(project_file)
            self.obj_list = self.params.obj_list
            self.seg_list = self.params.seg_list
            self.mean = np.array(self.params.mean)
            self.std  = np.array(self.params.std)
        else:
            # BDD100K defaults
            self.obj_list = ["car", "truck", "bus", "person", "bike", "motor", "tl", "ts", "block"]
            self.seg_list = ["background", "drivable", "lane"]
            self.mean = np.array([0.485, 0.456, 0.406])
            self.std  = np.array([0.229, 0.224, 0.225])
            self.params = None

        self.use_onnx = False
        self.pytorch_model = None
        self._features_cache = None  # Last feature map for GradCAM

        # Provider priority: TensorRT → CUDA → CPU
        if use_cuda:
            providers = [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]

        # ── ONNX Path Resolution ──────────────────────────────────────────
        fp16_path = model_path.replace(".onnx", "_fp16.onnx")
        active_path = fp16_path if os.path.exists(fp16_path) else model_path

        if not os.path.exists(active_path):
            print(f"[PerceptionModule] ONNX not found at {active_path}. Attempting auto-conversion...")
            try:
                import subprocess
                root = os.path.join(os.path.dirname(__file__), "..")
                subprocess.run(
                    [sys.executable, os.path.join(root, "setup_models.py")],
                    capture_output=True, timeout=120
                )
            except Exception as e:
                print(f"[PerceptionModule] Auto-conversion failed: {e}")

        # ── Try ONNX Runtime ──────────────────────────────────────────────
        if os.path.exists(active_path):
            try:
                import onnxruntime as ort
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess_options.inter_op_num_threads = 2
                sess_options.intra_op_num_threads = 4

                self.session = ort.InferenceSession(active_path, sess_options, providers=providers)
                self.input_name = self.session.get_inputs()[0].name
                # Get expected input shape from model
                input_shape = self.session.get_inputs()[0].shape  # [1, 3, H, W]
                self.model_h = input_shape[2] if isinstance(input_shape[2], int) else 384
                self.model_w = input_shape[3] if isinstance(input_shape[3], int) else 640
                self.use_onnx = True
                print(f"[PerceptionModule] ✓ ONNX Runtime loaded → {active_path} ({self.model_h}×{self.model_w})")
            except Exception as e:
                print(f"[PerceptionModule] ONNX loading failed ({e}). Switching to PyTorch.")
        else:
            print("[PerceptionModule] No ONNX model available.")

        # ── PyTorch Fallback ──────────────────────────────────────────────
        if not self.use_onnx:
            self.model_h, self.model_w = 384, 640
            pth_path = "weights/hybridnets.pth"
            if os.path.exists(pth_path) and HYBRIDNETS_AVAILABLE:
                try:
                    import torch
                    from backbone import HybridNetsBackbone
                    self.pytorch_model = HybridNetsBackbone(
                        num_classes=len(self.obj_list),
                        compound_coef=3,
                        seg_classes=len(self.seg_list),
                    )
                    ckpt = torch.load(pth_path, map_location="cuda" if use_cuda else "cpu")
                    state = ckpt.get("model", ckpt)
                    # Strip 'model.' prefix if present
                    from collections import OrderedDict
                    new_state = OrderedDict()
                    for k, v in state.items():
                        new_state[k[6:] if k.startswith("model.") else k] = v
                    self.pytorch_model.load_state_dict(new_state, strict=False)
                    self.pytorch_model.eval()
                    if use_cuda:
                        self.pytorch_model.cuda()
                    print(f"[PerceptionModule] ✓ PyTorch model loaded → {pth_path}")
                except Exception as e:
                    print(f"[PerceptionModule] PyTorch load failed: {e}. Using OpenCV fallback.")
                    self.pytorch_model = None
            else:
                print("[PerceptionModule] No weights available. Using OpenCV-based fallback detector.")

        # Post-processing helpers (BBoxTransform, ClipBoxes)
        if HYBRIDNETS_AVAILABLE:
            self.regressBoxes = BBoxTransform()
            self.clipBoxes = ClipBoxes()
        else:
            self.regressBoxes = None
            self.clipBoxes = None

        # Warmup
        if self.use_onnx:
            self._warmup()

        print("[PerceptionModule] Ready.")

    # ─────────────────────────────────────────────────────────────────────
    def _warmup(self, n: int = 3):
        """Run dummy inference to warm up ONNX Runtime."""
        print("[PerceptionModule] Warming up ONNX engine...")
        dummy = np.random.randn(1, 3, self.model_h, self.model_w).astype(np.float32)
        for _ in range(n):
            try:
                self.session.run(None, {self.input_name: dummy})
            except Exception:
                break

    # ─────────────────────────────────────────────────────────────────────
    def _preprocess(self, frame: np.ndarray, resolution: int = 640):
        """Resize, letterbox, normalize, and batch a BGR frame."""
        h0, w0 = frame.shape[:2]
        r = resolution / max(h0, w0)
        resized = cv2.resize(frame, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_AREA)
        h, w = resized.shape[:2]

        if HYBRIDNETS_AVAILABLE:
            (padded, _), ratio, pad = letterbox((resized, None), resolution, auto=True, scaleup=False)
        else:
            # Manual letterboxing
            pad_h = (resolution - h) // 2
            pad_w = (resolution - w) // 2
            padded = cv2.copyMakeBorder(resized, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=114)
            ratio = (h / h0, w / w0)
            pad = (pad_h, pad_w)

        shape_info = ((h0, w0), ((h / h0, w / w0), pad))

        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb / 255.0 - self.mean) / self.std
        tensor = rgb.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, H, W)

        return tensor.astype(np.float32), shape_info

    # ─────────────────────────────────────────────────────────────────────
    def _fallback_detect(self, frame: np.ndarray):
        """
        OpenCV-based fallback detection when no DL model is available.
        Uses background subtraction + contour detection for moving objects.
        Returns same dict schema as the DL path.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = frame.shape[:2]
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500 or area > w * h * 0.5:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            detections.append({
                "bbox": [x, y, x + bw, y + bh],
                "class": "car",
                "score": 0.5,
            })

        # Simple lane mask via Canny on bottom half
        bottom = frame[h // 2:, :]
        edges = cv2.Canny(cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY), 50, 150)
        lane_mask = np.zeros((h, w), dtype=np.uint8)
        lane_mask[h // 2:, :] = edges
        drivable_mask = np.zeros((h, w), dtype=np.uint8)
        drivable_mask[h * 2 // 3:, w // 4: w * 3 // 4] = 255

        return {
            "detections": detections[:10],
            "lane_mask": lane_mask,
            "drivable_mask": drivable_mask,
            "features": [],
        }

    # ─────────────────────────────────────────────────────────────────────
    def run(
        self,
        frame: np.ndarray,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.3,
        resolution: int = 640,
        frame_idx: int = 0,
        debug: bool = False,
    ) -> dict:
        """
        Run full perception on a single BGR frame.

        Returns
        -------
        dict with keys:
            detections    : list of {bbox, class, score}
            lane_mask     : np.uint8 (H, W)
            drivable_mask : np.uint8 (H, W)
            features      : list[np.ndarray] — feature maps for GradCAM
        """
        self.timer.start()

        # ── No model available ─────────────────────────────────────────
        if not self.use_onnx and self.pytorch_model is None:
            result = self._fallback_detect(frame)
            self.timer.stop()
            return result

        # ── Preprocess ────────────────────────────────────────────────
        input_tensor, shape_info = self._preprocess(frame, resolution)
        h0, w0 = shape_info[0]

        # ── Inference ─────────────────────────────────────────────────
        self.inf_timer.start()
        if self.use_onnx:
            outputs = self.session.run(None, {self.input_name: input_tensor})
            # HybridNets ONNX output: [features, regression, classification, seg]
            if len(outputs) == 4:
                feat_np, regression, classification, seg = outputs
            elif len(outputs) == 3:
                feat_np = None
                regression, classification, seg = outputs
            else:
                self.timer.stop()
                return self._fallback_detect(frame)
        else:
            import torch
            with torch.no_grad():
                device = next(self.pytorch_model.parameters()).device
                x = torch.from_numpy(input_tensor).to(device)
                feat_np, regression, classification, anchors, seg = self.pytorch_model(x)
                feat_np      = feat_np.cpu().numpy() if feat_np is not None else None
                regression    = regression.cpu().numpy()
                classification = classification.cpu().numpy()
                seg           = seg.cpu().numpy()
        self.inf_timer.stop()

        # Cache features for GradCAM
        self._features_cache = feat_np

        # ── Postprocess Segmentation ──────────────────────────────────
        seg_mask = np.argmax(seg[0], axis=0)

        # Unpad
        pad = shape_info[1][1]
        if HYBRIDNETS_AVAILABLE:
            pad_h = int(pad[1]) if isinstance(pad, (list, tuple)) and len(pad) > 1 else 0
            pad_w = int(pad[0]) if isinstance(pad, (list, tuple)) else 0
        else:
            pad_h = pad[0] if isinstance(pad, (list, tuple)) else 0
            pad_w = pad[1] if isinstance(pad, (list, tuple)) and len(pad) > 1 else 0

        seg_h, seg_w = seg_mask.shape
        if pad_h > 0:
            seg_mask = seg_mask[pad_h: seg_h - pad_h, :]
        if pad_w > 0:
            seg_mask = seg_mask[:, pad_w: seg_w - pad_w]
        seg_mask = cv2.resize(seg_mask.astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST)

        drivable_mask = (seg_mask == 1).astype(np.uint8) * 255
        lane_mask     = (seg_mask == 2).astype(np.uint8) * 255

        # ── Postprocess Detections ────────────────────────────────────
        detections = []
        try:
            import torch
            regression_t    = torch.from_numpy(regression)
            classification_t = torch.from_numpy(classification)

            if HYBRIDNETS_AVAILABLE and self.regressBoxes is not None:
                out = postprocess(
                    torch.from_numpy(input_tensor), None,
                    regression_t, classification_t,
                    self.regressBoxes, self.clipBoxes,
                    conf_thresh, iou_thresh,
                )
                rois      = out[0]["rois"]
                class_ids = out[0]["class_ids"]
                scores    = out[0]["scores"]

                if len(rois) > 0:
                    rois = scale_coords(
                        (self.model_h, self.model_w), rois, shape_info[0], shape_info[1]
                    )
                    for i in range(len(rois)):
                        x1, y1, x2, y2 = rois[i].astype(int)
                        # Clamp to frame bounds
                        x1 = max(0, min(x1, w0)); x2 = max(0, min(x2, w0))
                        y1 = max(0, min(y1, h0)); y2 = max(0, min(y2, h0))
                        if x2 - x1 < 5 or y2 - y1 < 5:
                            continue
                        detections.append({
                            "bbox":  [int(x1), int(y1), int(x2), int(y2)],
                            "class": self.obj_list[int(class_ids[i])],
                            "score": float(scores[i]),
                        })
        except Exception as e:
            if debug:
                print(f"[PerceptionModule] Detection postprocess error (frame {frame_idx}): {e}")

        if debug and frame_idx % 30 == 0:
            print(f"[PerceptionModule] Frame {frame_idx}: {len(detections)} detections")

        # Build feature list for GradCAM
        features = [feat_np] if feat_np is not None else []

        self.timer.stop()
        return {
            "detections":    detections,
            "lane_mask":     lane_mask,
            "drivable_mask": drivable_mask,
            "features":      features,
        }

    # ─────────────────────────────────────────────────────────────────────
    @property
    def last_features(self):
        """Return cached feature maps (for external GradCAM call)."""
        return self._features_cache
