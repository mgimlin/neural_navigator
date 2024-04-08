import cv2
import supervision as sv
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

    specificClassID = 67 # change for a different class you want the info for

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")

        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)

        specificClass = [detection for detection in detections if detection[3] == specificClassID]
        for detection in specificClass:
            specificClassBB = detection[0]

            x1, y1, x2, y2 = specificClassBB

            xCenter = (x1 + x2) / 2
            yCenter = (y1 + y2) / 2

            print(f"Center Coordinates: ({xCenter}, {yCenter})")

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