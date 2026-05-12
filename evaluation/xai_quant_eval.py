import numpy as np
import cv2

def calculate_heatmap_iou(heatmap, gt_bboxes, threshold=0.5):
    """
    Quantifies XAI by checking if the highest energy in the Grad-CAM heatmap 
    overlaps with the ground truth bounding boxes of relevant obstacles.
    """
    if len(gt_bboxes) == 0:
        return 0.0
    
    # Binary mask from heatmap
    binary_heatmap = (heatmap > (heatmap.max() * threshold)).astype(np.uint8)
    
    # Create mask for all GT bboxes
    h, w = heatmap.shape[:2]
    gt_mask = np.zeros((h, w), dtype=np.uint8)
    for box in gt_bboxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(gt_mask, (x1, y1), (x2, y2), 255, -1)
        
    intersection = np.logical_and(binary_heatmap, gt_mask).sum()
    union = np.logical_or(binary_heatmap, gt_mask).sum()
    
    iou = intersection / union if union > 0 else 0.0
    return iou

def validate_xai_batch(logs):
    """
    Evaluates a batch of system logs for XAI reliability.
    """
    results = []
    for log in logs:
        heatmap = log.get("heatmap") # Normalized 0-1
        gt_boxes = [obj["bbox"] for obj in log.get("objects", [])]
        
        iou = calculate_heatmap_iou(heatmap, gt_boxes)
        results.append(iou)
        
    mean_iou = np.mean(results)
    reliability = "HIGH" if mean_iou > 0.6 else "LOW (Hallucination risk)"
    
    print(f"XAI Quantified Reliability: {mean_iou:.4f} IoU -> [{reliability}]")
    return mean_iou
