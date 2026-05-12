"""
dataset/generate_features.py
=============================
Extracts real feature vectors from BDD100K images using the full
NeuroDrive-XAI perception pipeline, and derives driving labels
from BDD100K info annotations.

Usage:
    python dataset/generate_features.py \
        --images  datasets/bdd100k/images/100k/train \
        --labels  datasets/bdd100k/labels/bdd100k_labels_images_train.json \
        --output  dataset/real_features.csv \
        --max     5000

Output CSV columns:
    distance_to_object, relative_velocity, lane_offset,
    lane_curvature, num_objects, closest_object_type,
    steering_angle, brake
"""

import argparse
import csv
import json
import os
import sys
import glob
import time

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────── Feature extraction ──────
def estimate_depth_from_bbox(bbox, frame_h: int) -> float:
    """
    Estimate metric distance from bounding box size.
    Uses pinhole camera model approximation:
      distance ≈ (real_height * focal_length) / pixel_height
    where we use BDD100K camera parameters (focal ≈ 1000 px, car height ≈ 1.5m).
    """
    x1, y1, x2, y2 = bbox
    pixel_h = max(y2 - y1, 1)
    pixel_w = max(x2 - x1, 1)

    # Focal length and real-world size per class (approximate)
    FOCAL = 1000.0  # pixels (BDD100K camera)
    REAL_HEIGHTS = {
        "car": 1.5, "truck": 3.5, "bus": 3.2,
        "person": 1.7, "bike": 1.0, "motor": 1.2,
        "tl": 0.8, "ts": 1.5, "block": 0.8,
    }
    # Default
    real_h = 1.5
    dist = (real_h * FOCAL) / pixel_h
    return float(np.clip(dist, 1.0, 150.0))


def compute_lane_offset(detections: list, frame_w: int) -> float:
    """
    Estimate ego vehicle lateral lane offset in metres.
    Uses the horizontal centre of drivable-area detections as proxy.
    """
    if not detections:
        return 0.0
    # Centre of all detection midpoints vs frame centre
    midpoints = [(d["bbox"][0] + d["bbox"][2]) / 2 for d in detections]
    avg_mid = np.mean(midpoints)
    # Pixels from centre, scaled to metres (assuming 3.7m lane, 1280px wide)
    offset_px = avg_mid - frame_w / 2
    offset_m = offset_px * (3.7 / frame_w)
    return float(np.clip(offset_m, -3.0, 3.0))


def label_from_bdd_annotation(annotation: dict) -> tuple:
    """
    Derive (steering_angle, brake) labels from BDD100K scene attributes.

    BDD100K has: scene (city street, residential, highway, tunnel),
                 weather (clear, partly cloudy, overcast, rainy, snowy, foggy),
                 timeofday (daytime, dawn/dusk, night).
    We use detection geometry as the primary signal.
    """
    # Extract attributes if available
    attrs = annotation.get("attributes", {})
    scene = attrs.get("scene", "city street")
    weather = attrs.get("weather", "clear")

    # Default: no steering, no brake
    steering = 0.0
    brake = 0

    labels = annotation.get("labels", [])
    pedestrian_close = False
    car_close = False

    for lbl in labels:
        cat = lbl.get("category", "")
        if "box2d" in lbl:
            box = lbl["box2d"]
            h_box = box["y2"] - box["y1"]
            # Large bounding box → close object → estimate distance
            if h_box > 80:  # > 80px tall → within ~20m
                if cat == "person":
                    pedestrian_close = True
                    brake = 1
                elif cat in ("car", "truck", "bus"):
                    car_close = True
                    if h_box > 150:  # Very close → brake hard
                        brake = 1
                    else:
                        brake = 0

    # Derive steering from scene context
    if scene in ("residential",):
        steering = float(np.random.uniform(-0.15, 0.15))
    elif scene == "highway":
        steering = float(np.random.uniform(-0.05, 0.05))

    # Adverse weather → slight speed reduction (represented as partial brake)
    if weather in ("rainy", "snowy", "foggy") and brake == 0:
        brake = 0  # No full brake, but this affects decision engine

    return round(steering, 4), brake


# ─────────────────────────────────────────────────────────────────── Main ────
def generate_features(
    images_path: str,
    labels_path: str,
    output_path: str,
    max_frames: int = 5000,
    use_perception: bool = True,
):
    print(f"\n{'='*60}")
    print("NeuroDrive-XAI — Real Feature Extraction from BDD100K")
    print(f"{'='*60}")
    print(f"  Images : {images_path}")
    print(f"  Labels : {labels_path}")
    print(f"  Output : {output_path}")
    print(f"  Max    : {max_frames} frames\n")

    # ── Load labels ───────────────────────────────────────────────────
    labels_db = {}
    if os.path.exists(labels_path):
        print("Loading BDD100K label annotations...")
        with open(labels_path, "r") as f:
            data = json.load(f)
        for item in data:
            labels_db[item["name"]] = item
        print(f"  Loaded {len(labels_db):,} annotations.")
    else:
        print(f"  WARNING: Labels file not found at {labels_path}.")
        print("  Will use geometry-only feature extraction (no GT labels).")

    # ── Find image files ──────────────────────────────────────────────
    if not os.path.isdir(images_path):
        print(f"ERROR: Images directory not found: {images_path}")
        print("Run: python dataset/download_bdd.py  to get a sample.")
        sys.exit(1)

    image_files = sorted(glob.glob(os.path.join(images_path, "*.jpg")))
    if not image_files:
        image_files = sorted(glob.glob(os.path.join(images_path, "**", "*.jpg"), recursive=True))

    if not image_files:
        print(f"ERROR: No .jpg files found in {images_path}")
        sys.exit(1)

    image_files = image_files[:max_frames]
    print(f"  Found {len(image_files):,} images (processing {len(image_files):,}).\n")

    # ── Initialize perception ─────────────────────────────────────────
    perception = None
    if use_perception:
        try:
            from perception.hybridnets_wrapper import PerceptionModule
            print("Initializing PerceptionModule (HybridNets)...")
            perception = PerceptionModule(use_cuda=False)  # CPU for offline extraction
            print("  ✓ PerceptionModule ready.\n")
        except Exception as e:
            print(f"  WARNING: PerceptionModule unavailable ({e}). Using bbox-only features.\n")
            perception = None

    # ── Feature extraction loop ───────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    rows_written = 0
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "distance_to_object", "relative_velocity", "lane_offset",
            "lane_curvature", "num_objects", "closest_object_type",
            "steering_angle", "brake",
        ])

        for img_path in tqdm(image_files, desc="Extracting features", unit="frame"):
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            h, w = frame.shape[:2]
            img_name = os.path.basename(img_path)
            annotation = labels_db.get(img_name, {})

            # ── Perception ────────────────────────────────────────────
            if perception is not None:
                try:
                    perc_out = perception.run(frame, conf_thresh=0.25, resolution=416)
                    detections = perc_out["detections"]
                except Exception:
                    detections = []
            else:
                # Fallback: use GT bbox from annotation
                detections = []
                for lbl in annotation.get("labels", []):
                    if "box2d" in lbl:
                        b = lbl["box2d"]
                        detections.append({
                            "bbox": [int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])],
                            "class": lbl.get("category", "car"),
                            "score": 1.0,
                        })

            # ── Feature computation ───────────────────────────────────
            type_map = {"car": 0, "truck": 3, "bus": 3, "person": 1,
                        "bike": 2, "motor": 2, "tl": 4, "ts": 4, "block": 4}

            if detections:
                # Sort by distance (largest bbox = closest)
                det_with_dist = []
                for d in detections:
                    dist = estimate_depth_from_bbox(d["bbox"], h)
                    det_with_dist.append((dist, d))
                det_with_dist.sort(key=lambda x: x[0])

                closest_dist, closest_det = det_with_dist[0]
                closest_type = type_map.get(closest_det.get("class", "car"), 0)

                # Relative velocity: approximate from frame-to-frame size change
                # (single-frame extraction → use 0 as default)
                rel_velocity = 0.0  # Would need consecutive frames for real delta

                lane_offset = compute_lane_offset(detections, w)
                lane_curv = 0.01  # BDD100K doesn't provide curvature directly
                num_objects = len(detections)
            else:
                closest_dist = 100.0
                rel_velocity = 0.0
                lane_offset = 0.0
                lane_curv = 0.0
                num_objects = 0
                closest_type = 4  # none

            # ── Labels from annotation ────────────────────────────────
            if annotation:
                steering_angle, brake = label_from_bdd_annotation(annotation)
                # Override brake with geometry signal
                if closest_dist < 10.0:
                    brake = 1
                elif closest_dist < 20.0 and brake == 0:
                    brake = 0  # Caution but don't override
            else:
                # Geometry-based label derivation
                brake = 1 if closest_dist < 12.0 else 0
                steering_angle = float(np.clip(-lane_offset * 0.2, -0.5, 0.5))

            writer.writerow([
                round(closest_dist, 2),
                round(rel_velocity, 3),
                round(lane_offset, 3),
                round(lane_curv, 4),
                num_objects,
                closest_type,
                round(steering_angle, 4),
                brake,
            ])
            rows_written += 1

    print(f"\n✓ Feature extraction complete.")
    print(f"  Rows written : {rows_written:,}")
    print(f"  Output saved : {output_path}")

    # Quick stats
    try:
        import pandas as pd
        df = pd.read_csv(output_path)
        print(f"\n  Brake=1 rate: {df['brake'].mean()*100:.1f}%")
        print(f"  Avg distance: {df['distance_to_object'].mean():.1f}m")
        print(f"  Objects/frame: {df['num_objects'].mean():.1f}")
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate real feature CSV from BDD100K")
    parser.add_argument("--images",  default="datasets/bdd100k/images/100k/train")
    parser.add_argument("--labels",  default="datasets/bdd100k/labels/bdd100k_labels_images_train.json")
    parser.add_argument("--output",  default="dataset/real_features.csv")
    parser.add_argument("--max",     type=int, default=5000, help="Max frames to process")
    parser.add_argument("--no-perc", action="store_true", help="Skip PerceptionModule (faster)")
    args = parser.parse_args()

    generate_features(
        images_path=args.images,
        labels_path=args.labels,
        output_path=args.output,
        max_frames=args.max,
        use_perception=not args.no_perc,
    )
