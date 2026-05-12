# 🛠️ Fix Guide: Missing Models & DLLs

If you encountered errors like `NO_SUCHFILE` or `cuBLAS missing`, follow these two steps to fix your environment:

### 1. Run the Model Setup Script
You need to convert the PyTorch models to the optimized ONNX format for your hardware. Run this command once:
```powershell
python setup_models.py
```
This will create `weights/hybridnets.onnx` and `weights/hybridnets_fp16.onnx`.

### 2. Provider Fallback (Automatic)
The code has been updated to automatically verify your GPU drivers:
- If **TensorRT** is missing DLLs (like `cublas64_12.dll`), it will automatically fall back to **Standard CUDA**.
- If no GPU is found, it will fall back to **CPU mode** so the program doesn't crash.

### 3. Updated Run Commands
```powershell
# Optimized high-speed version:
python -m core.executor --input demo/messy_drive.mp4

# Video generation version:
python main_pipeline.py --input demo/messy_drive.mp4 --output artifacts/fixed_demo.mp4
```
