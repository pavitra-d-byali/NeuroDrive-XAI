"""
NeuroDrive-XAI — Model Setup & ONNX Conversion
Run this ONCE before running main_pipeline.py or core.executor

Usage:
    python setup_models.py
"""

import os
import sys
import torch

# ── Dependency Check ─────────────────────────────────────────────────────────
try:
    import onnxruntime
    import onnxscript
except ImportError as e:
    print(f"\n[MISSING DEPENDENCY] {e}")
    print("Please install the required ONNX exporter tools:")
    print("pip install onnxruntime-gpu onnxscript onnx")
    # We continue, but this likely will fail shortly if converting.

# ─────────────────────────────────────────────────────────────────────────────
HYBRIDNETS_DIR = os.path.join(os.path.dirname(__file__), "HybridNets")
sys.path.insert(0, HYBRIDNETS_DIR)

from backbone import HybridNetsBackbone
from utils.utils import Params
from utils.constants import MULTICLASS_MODE

# ─────────────────────────────────────────────────────────────────────────────

WEIGHTS_DIR   = os.path.join(os.path.dirname(__file__), "weights")
PTH_PATH      = os.path.join(WEIGHTS_DIR, "hybridnets.pth")
ONNX_PATH     = os.path.join(WEIGHTS_DIR, "hybridnets.onnx")
PROJECT_FILE  = os.path.join(HYBRIDNETS_DIR, "projects", "bdd100k.yml")
INPUT_HEIGHT  = 384
INPUT_WIDTH   = 640


def run():
    print("=== NeuroDrive-XAI: Model Setup & ONNX Conversion ===\n")

    # 1. Validate weights exist
    if not os.path.exists(PTH_PATH):
        print(f"[ERROR] PyTorch weights not found: {PTH_PATH}")
        print("Please make sure 'weights/hybridnets.pth' is present in the project folder.")
        return False

    # 2. Already converted?
    if os.path.exists(ONNX_PATH):
        print(f"[OK] ONNX model already exists at: {ONNX_PATH}")
        print("Nothing to do. Run main_pipeline.py directly.")
        return True

    # 3. Load project params
    print(f"[1/4] Loading project config: {PROJECT_FILE}")
    if not os.path.exists(PROJECT_FILE):
        print(f"[ERROR] Project file not found: {PROJECT_FILE}")
        return False

    params = Params(PROJECT_FILE)

    # 4. Build model
    print(f"[2/4] Building HybridNets model (compound_coef=3)...")
    model = HybridNetsBackbone(
        num_classes   = len(params.obj_list),
        compound_coef = 3,
        ratios        = eval(params.anchors_ratios),
        scales        = eval(params.anchors_scales),
        seg_classes   = len(params.seg_list),
        backbone_name = None,
        seg_mode      = MULTICLASS_MODE,
        onnx_export   = True,
    )

    # 5. Load weights (strip optimizer state if needed)
    print(f"[3/4] Loading weights from: {PTH_PATH}")
    checkpoint = torch.load(PTH_PATH, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        from collections import OrderedDict
        state = OrderedDict((k[6:], v) for k, v in checkpoint['model'].items())
    else:
        state = checkpoint
    model.load_state_dict(state, strict=False)
    model.eval()

    # 6. Export to ONNX
    print(f"[4/4] Exporting to ONNX: {ONNX_PATH}")
    dummy_input = torch.randn(1, 3, INPUT_HEIGHT, INPUT_WIDTH)
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        opset_version  = 11,
        input_names    = ['input'],
        output_names   = ['regression', 'classification', 'segmentation'],
        do_constant_folding = True,
    )

    print(f"\n✅ ONNX model saved → {ONNX_PATH}")
    print("You can now run:\n  python main_pipeline.py --input demo/messy_drive.mp4 --output artifacts/output_demo.mp4\n")
    return True


if __name__ == "__main__":
    success = run()
    sys.exit(0 if success else 1)
