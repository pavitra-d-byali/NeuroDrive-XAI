import sys
import os
import cv2
import torch
import numpy as np
from torchvision import transforms

# Ensure HybridNets is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "HybridNets"))

from backbone import HybridNetsBackbone
from utils.utils import letterbox, scale_coords, postprocess, BBoxTransform, ClipBoxes, Params

class PerceptionModule:
    def __init__(self, weight_path="weights/hybridnets.pth", project_file="HybridNets/projects/bdd100k.yml", use_cuda=False):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        
        self.params = Params(project_file)
        self.obj_list = self.params.obj_list
        self.seg_list = self.params.seg_list
        
        # Load model
        print("Loading real HybridNets model...")
        weight = torch.load(weight_path, map_location=self.device)
        
        # Determine segmentation mode. For BDD100K it's MULTICLASS_MODE (1)
        # hybridnets.pth has 3 segmentation classes (background, lane, drivable)
        self.model = HybridNetsBackbone(compound_coef=3, num_classes=len(self.obj_list), 
                                        ratios=eval(self.params.anchors_ratios),
                                        scales=eval(self.params.anchors_scales), 
                                        seg_classes=len(self.seg_list), 
                                        backbone_name=None,
                                        seg_mode=1)
        
        # HybridNets weight dict sometimes is nested under 'model' or direct
        state_dict = weight if "model" not in weight else weight["model"]
        self.model.load_state_dict(state_dict, strict=False)
        self.model.requires_grad_(False)
        self.model.eval()
        self.model = self.model.to(self.device)
        
        self.normalize = transforms.Normalize(mean=self.params.mean, std=self.params.std)
        self.transform = transforms.Compose([transforms.ToTensor(), self.normalize])
        
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        print("HybridNets loaded.")

    def run(self, frame, conf_thresh=0.25, iou_thresh=0.3, debug=False, frame_idx=-1):
        if debug:
            conf_thresh = 0.15
        
        # Preprocess
        h0, w0 = frame.shape[:2]
        resized_shape = self.params.model['image_size']
        if isinstance(resized_shape, list): resized_shape = max(resized_shape)
        
        r = resized_shape / max(h0, w0)
        input_img = cv2.resize(frame, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_AREA)
        h, w = input_img.shape[:2]
        
        (input_img, _), ratio, pad = letterbox((input_img, None), resized_shape, auto=True, scaleup=False)
        shape_info = ((h0, w0), ((h / h0, w / w0), pad))
        
        # RGB Conversion and Tensor
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        x = self.transform(input_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features, regression, classification, anchors, seg = self.model(x)
            
            # Seg mask output: shape is (B, C, W, H) -> (B, W, H)
            _, seg_mask = torch.max(seg, 1)
            seg_mask_arr = seg_mask[0].squeeze().cpu().numpy()
            
            # Unpad segmentation mask
            pad_h, pad_w = int(shape_info[1][1][1]), int(shape_info[1][1][0])
            seg_mask_arr = seg_mask_arr[pad_h:seg_mask_arr.shape[0]-pad_h, pad_w:seg_mask_arr.shape[1]-pad_w]
            seg_mask_arr = cv2.resize(seg_mask_arr, dsize=(w0, h0), interpolation=cv2.INTER_NEAREST)
            
            if debug and frame_idx < 3:
                print(f"\n[DEBUG] Frame {frame_idx} Raw Model Outputs BEFORE filtering:")
                print(f"Regression tensor shape: {regression.shape}")
                print(f"Classification tensor shape: {classification.shape}")
                print(f"Max class score (confidence): {classification.max().item():.4f}")

            # Detections (Filtered)
            out = postprocess(x, anchors, regression, classification, 
                              self.regressBoxes, self.clipBoxes, conf_thresh, iou_thresh)
            
            rois = out[0]['rois']
            class_ids = out[0]['class_ids']
            scores = out[0]['scores']
            
            detections = []
            if len(rois) > 0:
                rois = scale_coords(x.shape[2:], rois, shape_info[0], shape_info[1])
                for i in range(len(rois)):
                    x1, y1, x2, y2 = rois[i].astype(int)
                    cid = int(class_ids[i])
                    obj_class = self.obj_list[cid] if cid < len(self.obj_list) else str(cid)
                    detections.append({
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "class": obj_class,
                        "score": float(scores[i])
                    })

            # Raw Detections (threshold 0.01)
            raw_detections = []
            if debug:
                out_raw = postprocess(x, anchors, regression, classification, 
                                      self.regressBoxes, self.clipBoxes, 0.01, iou_thresh)
                rois_raw = out_raw[0]['rois']
                class_ids_raw = out_raw[0]['class_ids']
                scores_raw = out_raw[0]['scores']
                
                if len(rois_raw) > 0:
                    rois_raw = scale_coords(x.shape[2:], rois_raw, shape_info[0], shape_info[1])
                    for i in range(len(rois_raw)):
                        x1, y1, x2, y2 = rois_raw[i].astype(int)
                        cid = int(class_ids_raw[i])
                        obj_class = self.obj_list[cid] if cid < len(self.obj_list) else str(cid)
                        raw_detections.append({
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "class": obj_class,
                            "score": float(scores_raw[i])
                        })
            
            # For HybridNets bdd100k defaults: class 1 is drivable area, class 2 is lane line
            drivable_mask = (seg_mask_arr == 1).astype(np.uint8) * 255
            lane_mask = (seg_mask_arr == 2).astype(np.uint8) * 255
            
            return {
                "detections": detections,
                "raw_detections": raw_detections if debug else [],
                "lane_mask": lane_mask,
                "drivable_mask": drivable_mask,
                "features": features
            }
