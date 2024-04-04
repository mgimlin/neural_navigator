from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')

results = model.track(source="0", show=True)  # Tracking with default tracker
