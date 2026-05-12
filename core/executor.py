import multiprocessing as mp
import time
import cv2
import queue
import numpy as np
from perception.hybridnets_wrapper import PerceptionModule
from planning.decision_engine import DecisionEngine
from control.inference import HybridControlInference
from utils.profiler import PerfTimer

class AsyncAVSystem:
    """
    Parallel Multiprocessing Executor for NeuroDrive-XAI.
    Separates Perception (GPU heavy) from Planning/Control (CPU heavy).
    """
    def __init__(self, video_path):
        self.video_path = video_path
        self.frame_queue = mp.Queue(maxsize=2)  # Backpressure: limit backlog
        self.result_queue = mp.Queue()
        self.stop_event = mp.Event()
        
        self.perf = PerfTimer("End-to-End")

    def perception_process(self, input_q, output_q, stop_evt):
        """Dedicated process for GPU perception."""
        perception = PerceptionModule(use_cuda=True)
        print("[Process-Perception] Started.")
        
        while not stop_evt.is_set():
            try:
                frame_data = input_q.get(timeout=1.0)
                if frame_data is None: break
                
                idx, frame = frame_data
                results = perception.run(frame)
                output_q.put((idx, results, frame))
            except queue.Empty:
                continue

    def run(self):
        # Start processes
        p_proc = mp.Process(target=self.perception_process, 
                            args=(self.frame_queue, self.result_queue, self.stop_event))
        p_proc.start()
        
        cap = cv2.VideoCapture(self.video_path)
        decision_engine = DecisionEngine()
        
        frame_idx = 0
        print("[Main-Loop] Starting...")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # Push to perception (non-blocking if queue has space)
                try:
                    self.frame_queue.put_nowait((frame_idx, frame))
                except queue.Full:
                    # Drop frame to maintain real-time (Point 5/6)
                    print(f"Skipping frame {frame_idx} due to backpressure")
                    continue
                
                # Check for results (non-blocking)
                try:
                    res_idx, p_res, orig_frame = self.result_queue.get_nowait()
                    
                    # Run Planning & Control (Point 4)
                    decision = decision_engine.decide(p_res)
                    print(f"Frame {res_idx}: Action -> {decision['action']}")
                    
                except queue.Empty:
                    pass # Continue capturing
                
                frame_idx += 1
                # Process full video
        finally:
            self.stop_event.set()
            p_proc.join()
            cap.release()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", "--input", "-i", type=str, default="demo/messy_drive.mp4", help="Path to input video")
    args = parser.parse_args()
    
    av = AsyncAVSystem(args.video)
    av.run()
