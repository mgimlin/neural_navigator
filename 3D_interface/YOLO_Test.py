import cv2
import supervision as sv
import numpy as np
from ultralytics import YOLO

def main():
    model = YOLO("yolov8n-seg.pt")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1440)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")

        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
       
        labels = [f"{model.model.names[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, _, _ in detections]
        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        cv2.imshow('YOLOv8', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()