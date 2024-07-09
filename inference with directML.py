#Integrate the optimized ONNX model into a real-time application using DirectML.
import cv2
import numpy as np
import onnxruntime as ort

# Load the optimized ONNX model
ort_session = ort.InferenceSession("optimized_model.onnx", providers=['DmlExecutionProvider'])

# Define the preprocessing function
def preprocess(image):
    image = cv2.resize(image, (224, 224))
    image = image.transpose(2, 0, 1)
    image = image.astype(np.float32)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Define the postprocessing function
def postprocess(outputs, threshold=0.5):
    boxes, scores, labels = outputs
    boxes = boxes[0]
    scores = scores[0]
    labels = labels[0]
    indices = np.where(scores > threshold)[0]
    boxes = boxes[indices]
    scores = scores[indices]
    labels = labels[indices]
    return boxes, scores, labels

# Real-time object detection using webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    input_image = preprocess(frame)
    ort_inputs = {ort_session.get_inputs()[0].name: input_image}
    ort_outs = ort_session.run(None, ort_inputs)
    
    boxes, scores, labels = postprocess(ort_outs)
    
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{label}: {score:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
