import os
from ultralytics import YOLO
import cv2

# -------------------------------
# 1️⃣ Dataset paths
# -------------------------------
dataset_path = "data\DetectPot_AUG_Split\DetectPot_AUG_Split"
data_yaml = "data\PotholeDetection.yaml"  # your existing YAML

# -------------------------------
# 2️⃣ Train YOLOv8
# -------------------------------
model = YOLO("yolov8n.pt")  # small pretrained YOLOv8 model
print("Starting training...")
model.train(data=data_yaml, epochs=50, imgsz=640)
print("Training finished.")

# -------------------------------
# 3️⃣ Live detection
# -------------------------------
trained_model_path = "runs/train/exp/weights/best.pt"
model = YOLO(trained_model_path)

cap = cv2.VideoCapture(0)  # webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    # Draw bounding boxes
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        for box, conf, cls in zip(boxes, confidences, classes):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(cls)]
            color = (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Pothole Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
