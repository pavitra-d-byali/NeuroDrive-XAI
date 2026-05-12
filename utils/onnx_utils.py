import torch
import onnx
import os
from onnxconverter_common import float16

def convert_to_onnx(model, dummy_input, save_path, input_names=["input"], output_names=["output"]):
    """
    Converts PyTorch model to ONNX with standard optimizations.
    """
    print(f"Converting to ONNX: {save_path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        save_path, 
        export_params=True, 
        opset_version=13, 
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={input_names[0]: {0: 'batch_size'}}
    )
    
    # Check the model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX conversion successful.")

def optimize_fp16(onnx_path, save_path):
    """
    Applies FP16 quantization to an ONNX model for GPU acceleration.
    """
    print(f"Applying FP16 optimization: {save_path}...")
    model_onnx = onnx.load(onnx_path)
    model_fp16 = float16.convert_float_to_float16(model_onnx)
    onnx.save(model_fp16, save_path)
    print("FP16 optimization successful.")

if __name__ == "__main__":
    # Example for HybridNets or MLP
    # This would be called by a training/export script
    pass
