import cv2
import torch

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or 'yolov5m', 'yolov5l', 'yolov5x' for different sizes

# Open video capture
cap = cv2.VideoCapture(0)  # 0 for default camera, or replace with video file path

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model(frame)

    # Render results on the frame
    frame_with_boxes = results.render()[0]

    # Display the resulting frame
    cv2.imshow('YOLOv5 Object Detection', frame_with_boxes)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
