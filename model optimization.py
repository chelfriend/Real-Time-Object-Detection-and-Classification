#Convert the trained PyTorch model to ONNX format and optimize it.
import torch.onnx
import onnx
import onnxruntime

# Load the trained PyTorch model
model.load_state_dict(torch.load('fasterrcnn_resnet50_fpn.pth'))
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 224, 224).to(device)

# Export the model
torch.onnx.export(model, dummy_input, "model.onnx", opset_version=11)

# Optimize the ONNX model using ONNX Runtime
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
optimized_model = onnx.optimizer.optimize(onnx_model, passes=["eliminate_deadend"])

# Save the optimized model
onnx.save(optimized_model, "optimized_model.onnx")
