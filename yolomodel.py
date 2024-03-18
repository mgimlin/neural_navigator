from ultralytics import YOLO

model = YOLO('yolov8n-seg.yaml')

results = results = model.train(
   data='data.yaml',
   imgsz=640,
   epochs=100,
   batch=8,
   name='yolov8n_custom')