"""
explainability/gradcam.py
=========================
XAI visualization for NeuroDrive-XAI.

Strategy:
  1. If PyTorch model is available → full GradCAM via pytorch-grad-cam
  2. If ONNX-only → Gradient-free attention proxy:
       - Sobel edge magnitude (structural attention)
       - Blended with detection bounding-box saliency
  This gives a *meaningful* visual explanation even without autograd.
"""

import cv2
import numpy as np


class PerceptionXAI:
    """
    Produces per-frame visual explanations of perception decisions.
    Works regardless of whether PyTorch or ONNX Runtime is being used.
    """

    def __init__(self, perception_module):
        self.perception_module = perception_module
        self._gradcam = None

        # Try to set up full GradCAM if PyTorch model is available
        if getattr(perception_module, "pytorch_model", None) is not None:
            try:
                from pytorch_grad_cam import GradCAM
                from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

                model = perception_module.pytorch_model

                # Identify target layers for EfficientNet backbone
                target_layers = self._find_target_layers(model)
                if target_layers:
                    self._gradcam = GradCAM(model=model, target_layers=target_layers)
                    self._ClassifierOutputTarget = ClassifierOutputTarget
                    print("[PerceptionXAI] ✓ GradCAM initialized on PyTorch backbone.")
                else:
                    print("[PerceptionXAI] Could not find target layers. Using saliency proxy.")
            except ImportError:
                print("[PerceptionXAI] pytorch-grad-cam not installed. Using saliency proxy.")
            except Exception as e:
                print(f"[PerceptionXAI] GradCAM init failed ({e}). Using saliency proxy.")
        else:
            print("[PerceptionXAI] ONNX mode detected — using gradient-free saliency proxy.")

    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def _find_target_layers(model):
        """Find the last convolutional block in an EfficientNet backbone."""
        try:
            if hasattr(model, "encoder"):
                enc = model.encoder
                if hasattr(enc, "_blocks") and len(enc._blocks) > 0:
                    return [enc._blocks[-1]]
                if hasattr(enc, "blocks") and len(enc.blocks) > 0:
                    return [enc.blocks[-1]]
            # Generic: last Conv2d layer
            last_conv = None
            for m in model.modules():
                if isinstance(m, __import__("torch").nn.Conv2d):
                    last_conv = m
            return [last_conv] if last_conv else []
        except Exception:
            return []

    # ─────────────────────────────────────────────────────────────────────
    def _saliency_proxy(self, frame: np.ndarray, detections: list = None) -> np.ndarray:
        """
        Gradient-free attention map built from:
          - Sobel edge magnitude (where structure matters)
          - Detection bounding-box response (where the model focused)
        Returns a float32 heatmap in [0, 1] with the same H×W as frame.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Sobel edge response
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        # Gaussian blur to smooth edges
        attention = cv2.GaussianBlur(edge_mag, (31, 31), 0)

        # Emphasise detected object regions
        if detections:
            det_layer = np.zeros((h, w), dtype=np.float32)
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                score = det.get("score", 0.5)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    det_layer[y1:y2, x1:x2] += score
            # Blend: 60% edges, 40% detections
            det_blurred = cv2.GaussianBlur(det_layer, (51, 51), 0)
            det_max = det_blurred.max()
            if det_max > 0:
                det_blurred /= det_max
            attention = 0.6 * attention + 0.4 * det_blurred * attention.max()

        # Normalize to [0, 1]
        a_max = attention.max()
        if a_max > 0:
            attention /= a_max

        return attention.astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────
    def explain_detection(
        self,
        frame: np.ndarray,
        input_tensor=None,
        target_category_id: int = None,
        detections: list = None,
    ) -> np.ndarray:
        """
        Generate an XAI heatmap for a given frame.

        Parameters
        ----------
        frame               : BGR frame (H, W, 3)
        input_tensor        : torch.Tensor (1, 3, H, W) — only needed for GradCAM
        target_category_id  : int — class index to explain (GradCAM only)
        detections          : list of detection dicts (used for saliency proxy)

        Returns
        -------
        heatmap : np.ndarray float32 (H, W) in [0, 1]
        """
        # ── Full GradCAM path ──────────────────────────────────────────
        if self._gradcam is not None and input_tensor is not None:
            try:
                import torch
                if isinstance(input_tensor, np.ndarray):
                    input_tensor = torch.from_numpy(input_tensor)
                if input_tensor.dim() == 3:
                    input_tensor = input_tensor.unsqueeze(0)

                targets = (
                    [self._ClassifierOutputTarget(target_category_id)]
                    if target_category_id is not None
                    else None
                )
                grayscale_cam = self._gradcam(input_tensor=input_tensor, targets=targets)
                heatmap = grayscale_cam[0]  # (H_feat, W_feat)
                heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
                return heatmap.astype(np.float32)
            except Exception as e:
                # Fall through to proxy
                pass

        # ── Gradient-free proxy path ───────────────────────────────────
        return self._saliency_proxy(frame, detections=detections)

    # ─────────────────────────────────────────────────────────────────────
    def explain_segmentation(
        self,
        frame: np.ndarray,
        input_tensor=None,
        detections: list = None,
    ) -> np.ndarray:
        """
        Explains the segmentation head (Drivable Space / Lanes).
        Delegates to explain_detection with no specific class target.
        """
        return self.explain_detection(
            frame, input_tensor=input_tensor, detections=detections
        )

    # ─────────────────────────────────────────────────────────────────────
    def render_heatmap_overlay(
        self,
        frame: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Blend a (H, W) float heatmap onto a BGR frame.
        Returns BGR frame with jet colormap overlay.
        """
        h, w = frame.shape[:2]
        hm_uint8 = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
        hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
        if hm_color.shape[:2] != (h, w):
            hm_color = cv2.resize(hm_color, (w, h))
        return cv2.addWeighted(frame, 1 - alpha, hm_color, alpha, 0)
