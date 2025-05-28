import onnxruntime as ort
import cv2
import numpy as np

# Load ONNX model
session = ort.InferenceSession("model.onnx")

# Load and preprocess test image
image = cv2.imread("test_image.png")
image = cv2.resize(image, (640, 640))
image = np.transpose(image, (2, 0, 1)).astype(np.float32)  # Convert to channel-first format
image = np.expand_dims(image, axis=0)

# Run inference
outputs = session.run(None, {session.get_inputs()[0].name: image})

print("Detection Output:", outputs)

# Run the script with:
# python run_inference.py