from deep_sort_realtime.deepsort_tracker import DeepSort

class ObjectTracker:
    def __init__(self):
        self.tracker = DeepSort(max_age=30, n_init=2, nms_max_overlap=1.0)
        
    def update(self, detections, frame):
        """
        detections: list of dicts {"bbox": [x1, y1, x2, y2], "class": str, "score": float}
        """
        bbs = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            w = x2 - x1
            h = y2 - y1
            conf = det["score"]
            cls = det["class"]
            bbs.append(([x1, y1, w, h], conf, cls))
            
        tracks = self.tracker.update_tracks(bbs, frame=frame)
        
        tracked_objects = []
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
                
            track_id = track.track_id
            ltrb = track.to_ltrb() # left, top, right, bottom
            cls = track.get_det_class()
            
            tracked_objects.append({
                "track_id": track_id,
                "bbox": [int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])],
                "class": cls
            })
            
        return tracked_objects
