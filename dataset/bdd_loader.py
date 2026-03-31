import cv2
import json
import os
import glob

class BDD100kLoader:
    def __init__(self, data_path, dataset_root="datasets/bdd100k"):
        self.data_path = data_path
        self.is_video = data_path.endswith(('.mp4', '.avi', '.mov'))
        
        if self.is_video:
            self.cap = cv2.VideoCapture(data_path)
            self.labels = {}
        else:
            self.images_path = os.path.join(dataset_root, "images", "100k", "train")
            self.labels_path = os.path.join(dataset_root, "labels", "bdd100k_labels_images_train.json")
            
            # If a direct image sequence directory is provided instead
            if os.path.isdir(data_path):
                self.images_path = data_path
                self.image_files = sorted(glob.glob(os.path.join(data_path, "*.jpg")))
            
            self.labels = {}
            if os.path.exists(self.labels_path):
                with open(self.labels_path, 'r') as f:
                    data = json.load(f)
                    for item in data:
                        self.labels[item['name']] = item

    def get_frames(self):
        if self.is_video:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                yield {
                    "frame": frame,
                    "objects": []
                }
            self.cap.release()
        else:
            for img_path in self.image_files:
                img_name = os.path.basename(img_path)
                frame = cv2.imread(img_path)
                objects = []
                
                if img_name in self.labels:
                    for label in self.labels[img_name].get('labels', []):
                        if 'box2d' in label:
                            box = label['box2d']
                            objects.append({
                                "type": label['category'],
                                "bbox": [
                                    int(box['x1']),
                                    int(box['y1']),
                                    int(box['x2']),
                                    int(box['y2'])
                                ]
                            })
                
                yield {
                    "frame": frame,
                    "objects": objects
                }
