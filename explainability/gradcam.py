import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

class ExplainabilityModule:
    def __init__(self, perception_module):
        self.perception_module = perception_module
        
        # Real HybridNets backbone hook.
        # HybridNets uses an EfficientNet backbone.
        # EfficientNet features come from encoder blocks.
        if hasattr(perception_module.model, 'encoder'):
            if hasattr(perception_module.model.encoder, '_blocks'):
                target_layers = [perception_module.model.encoder._blocks[-1]]
            else:
                target_layers = [perception_module.model.encoder.blocks[-1]]
        elif hasattr(perception_module.model, 'backbone'):
            target_layers = [perception_module.model.backbone]
        else:
            # Fallback if structure is different
            target_layers = [list(perception_module.model.children())[-1]]
            
        self.cam = GradCAM(model=perception_module.model, target_layers=target_layers)
            
    def generate_heatmap(self, frame, input_tensor=None):
        if input_tensor is None:
            # Should not happen in real mode, but just in case
            return np.zeros_like(frame)
            
        # Target category None means highest scoring category
        # Since GradCAM operates on a forward pass, we need to pass the same tensor input
        grayscale_cam = self.cam(input_tensor=input_tensor.unsqueeze(0), targets=None)
        
        grayscale_cam = grayscale_cam[0, :]
        frame_normalized = np.float32(frame) / 255.0
        
        # Resize cam to frame size
        grayscale_cam = cv2.resize(grayscale_cam, (frame.shape[1], frame.shape[0]))
        
        cam_image = show_cam_on_image(frame_normalized, grayscale_cam, use_rgb=False)
        return cam_image
