from ultralytics import YOLO
import cv2

print("Loading model...")
model = YOLO(r'e:\github\young\FlaskTrafficMonitor\yolov8n.pt')
print("Model loaded successfully")

print("Opening video...")
cap = cv2.VideoCapture(r'e:\github\young\FlaskTrafficMonitor\uploads\road_redio.mp4')
print("Video opened:", cap.isOpened())

ret, frame = cap.read()
if ret:
    print("Frame read successfully, shape:", frame.shape)
    print("Running inference...")
    results = model(frame, verbose=False, conf=0.35, classes=[2,5,7])
    if results and len(results) > 0:
        print("Detections:", len(results[0].boxes))
    else:
        print("No detections")
else:
    print("Failed to read frame")

cap.release()
print("Done")