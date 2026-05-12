"""
dataset/download_bdd.py
=======================
Download BDD100K sample dataset (100 validation images + label JSON).

The full BDD100K dataset requires registration at:
  https://bdd-data.berkeley.edu/

This script downloads the freely available BDD100K mini-sample
(100 validation images from Berkeley's public CDN) for immediate
pipeline testing without registration.

Usage:
    python dataset/download_bdd.py
    python dataset/download_bdd.py --full  # Instructions for full dataset
"""

import argparse
import json
import os
import sys
import urllib.request
import zipfile
import shutil
from pathlib import Path


SAMPLE_IMAGES_URL = (
    "https://dl.cv.ethz.ch/bdd100k/data/100k_images_val.zip"
)

SAMPLE_LABELS_URL = (
    "https://dl.cv.ethz.ch/bdd100k/data/bdd100k_labels_release.zip"
)

# Mirror (GitHub-hosted 100-image sample)
GITHUB_SAMPLE_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
    "bdd100k_val_sample.zip"
)

OUTPUT_DIR = Path("datasets/bdd100k")


def download_with_progress(url: str, dest: str):
    """Download a file with a simple progress indicator."""
    print(f"  Downloading: {url}")
    print(f"  → {dest}")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"\r  [{bar}] {pct:.1f}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, dest, reporthook)
        print()  # newline after progress bar
        return True
    except Exception as e:
        print(f"\n  ERROR: {e}")
        return False


def create_synthetic_sample(output_dir: Path):
    """
    Create a minimal BDD100K-compatible sample using synthetic data
    when download fails (e.g., no internet connection).

    Generates 100 synthetic driving-scene images with matching label JSON.
    """
    import numpy as np
    import cv2

    images_dir = output_dir / "images" / "100k" / "train"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    print("\n  Creating synthetic BDD100K-compatible sample (100 images)...")

    labels_data = []
    np.random.seed(42)

    scenes = ["city street", "residential", "highway"]
    weathers = ["clear", "partly cloudy", "rainy"]
    times = ["daytime", "night"]

    for i in range(100):
        fname = f"bdd_sample_{i:04d}.jpg"

        # Create a synthetic driving scene
        img = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Sky gradient
        for row in range(360):
            alpha = row / 360.0
            img[row, :] = [int(50 + 100 * alpha), int(100 + 80 * alpha), int(180 - 50 * alpha)]

        # Road
        road_pts = np.array([[0, 720], [350, 380], [930, 380], [1280, 720]], np.int32)
        cv2.fillPoly(img, [road_pts], (80, 80, 80))

        # Lane markings
        cv2.line(img, (640, 720), (640, 380), (255, 255, 255), 3)

        # Random vehicles
        car_labels = []
        num_cars = np.random.randint(1, 5)
        for j in range(num_cars):
            cx = np.random.randint(200, 1080)
            cy = np.random.randint(420, 680)
            cw = np.random.randint(60, 200)
            ch = np.random.randint(40, 120)
            x1, y1 = max(0, cx - cw // 2), max(0, cy - ch // 2)
            x2, y2 = min(1280, cx + cw // 2), min(720, cy + ch // 2)

            # Draw car
            car_color = (np.random.randint(100, 255),) * 3
            cv2.rectangle(img, (x1, y1), (x2, y2), car_color, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 2)

            car_labels.append({
                "id": j,
                "category": "car",
                "manualShape": True,
                "manualAttributes": True,
                "box2d": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                "attributes": {"occluded": False, "truncated": False, "trafficLightColor": "none"},
            })

        cv2.imwrite(str(images_dir / fname), img)

        labels_data.append({
            "name": fname,
            "url": f"http://sample/{fname}",
            "videoName": "bdd_sample",
            "frameIndex": i,
            "attributes": {
                "weather": weathers[i % len(weathers)],
                "scene":   scenes[i % len(scenes)],
                "timeofday": times[i % len(times)],
            },
            "labels": car_labels,
        })

    # Save labels JSON
    labels_path = labels_dir / "bdd100k_labels_images_train.json"
    with open(labels_path, "w") as f:
        json.dump(labels_data, f, indent=2)

    print(f"  ✓ Created 100 synthetic images → {images_dir}")
    print(f"  ✓ Created labels JSON         → {labels_path}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Show instructions for full BDD100K download")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic sample (no internet required)")
    args = parser.parse_args()

    if args.full:
        print("""
Full BDD100K Dataset Download Instructions
==========================================

1. Register at: https://bdd-data.berkeley.edu/
2. Download:
   - bdd100k_images_100k.zip     (~1.8 GB) → images
   - bdd100k_labels_release.zip  (~400 MB) → labels

3. Extract to:
   datasets/bdd100k/
   ├── images/
   │   └── 100k/
   │       ├── train/  (70,000 images)
   │       └── val/    (10,000 images)
   └── labels/
       └── bdd100k_labels_images_train.json

4. Run feature extraction:
   python dataset/generate_features.py --max 5000
""")
        return

    print("\nNeuroDrive-XAI — BDD100K Sample Setup")
    print("=" * 45)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        create_synthetic_sample(OUTPUT_DIR)
        print("\n✓ Synthetic BDD100K sample ready.")
        print("  Run: python dataset/generate_features.py --no-perc")
        return

    # Try downloading real sample first
    print("\nAttempting to download BDD100K mini-sample...")
    tmp_zip = str(OUTPUT_DIR / "bdd_sample.zip")

    success = False
    for url in [GITHUB_SAMPLE_URL, SAMPLE_IMAGES_URL]:
        if download_with_progress(url, tmp_zip):
            try:
                print("  Extracting...")
                with zipfile.ZipFile(tmp_zip, "r") as z:
                    z.extractall(str(OUTPUT_DIR))
                os.remove(tmp_zip)
                success = True
                break
            except Exception as e:
                print(f"  Extraction failed: {e}")

    if not success:
        print("\n  Download failed. Creating synthetic sample instead...")
        create_synthetic_sample(OUTPUT_DIR)

    # Create label stub if not present
    labels_dir = OUTPUT_DIR / "labels"
    labels_path = labels_dir / "bdd100k_labels_images_train.json"
    if not labels_path.exists():
        print("  Creating label stub...")
        labels_dir.mkdir(exist_ok=True)
        with open(labels_path, "w") as f:
            json.dump([], f)

    print("\n✓ BDD100K sample ready at:", OUTPUT_DIR)
    print("\nNext steps:")
    print("  python dataset/generate_features.py")
    print("  python decision/train.py --data dataset/real_features.csv")


if __name__ == "__main__":
    main()
