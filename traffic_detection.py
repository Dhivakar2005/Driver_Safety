from ultralytics import YOLO
import cv2

# Load pre-trained YOLOv8 model (can be trained on traffic signs)
model = YOLO("yolov8n.pt")  # Use yolov8n.pt for a small, fast model

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)

    # Draw detections on frame
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # bounding boxes
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for box, conf, cls in zip(boxes, confidences, classes):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(cls)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Traffic Light & Road Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
