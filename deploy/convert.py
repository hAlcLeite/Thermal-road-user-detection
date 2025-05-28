# This file will convert the PyTorch model into ONNX and then to TensorRT for faster inference.
import torch

# Load the trained model
# best_model.pt is just a placeholder
model = torch.load("best_model.pt")  
model.eval()

# Dummy input (adjust input size)
dummy_input = torch.randn(1, 3, 640, 640)  

# Convert to ONNX
torch.onnx.export(model, dummy_input, "model.onnx", opset_version=11)