import os
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
import torch
import cv2
import supervision as sv
from tqdm import tqdm
import numpy as np
from PIL import Image

class dino:
    def __init__(self, home_dir=os.getcwd()):
        self.home_dir = home_dir

    def load_dino_sam(self, input_sam_path="sam_vit_h_4b8939.pth", input_weights_path="groundingdino_swint_ogc.pth", input_config_path="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"):
        config_path = os.path.join(self.home_dir, input_config_path)
        weights_path = os.path.join(self.home_dir, input_weights_path)
        if not (os.path.isfile(config_path) and os.path.isfile(weights_path)):
            print("load_dino: dino input files do not exist")
            return
        self.model = Model(model_config_path=config_path, model_checkpoint_path=weights_path)

        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        SAM_ENCODER_VERSION = "vit_h"
        SAM_CHECKPOINT_PATH = os.path.join(self.home_dir, input_sam_path)
        if not os.path.isfile(SAM_CHECKPOINT_PATH):
            print("load_dino: sam input files do not exist")
            return
        self.sam = SamPredictor(sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE))

    def annotate_images(self, images_path, labels_path, classes):
        images = []
        annotations = []
        for filename in tqdm(os.listdir(images_path)):
            if not filename.startswith('.'):  # Ignore hidden files
                file_path = os.path.join(images_path, filename)
                self.filename = filename
                image, detections = self.annotate(file_path, classes)

                images.append(image)
                annotations.append(detections)



    def annotate(self, image_path, classes):
        BOX_TRESHOLD = 0.35
        TEXT_TRESHOLD = 0.25

        image = cv2.imread(image_path)

        # detect objects
        detections = self.model.predict_with_classes(
            image=image,
            classes=classes,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )

        return self.segment(image_path, detections, image, classes)

    def segment(self, image_path, detections, image, classes):
        self.sam.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        result_masks = []
        for box in detections.xyxy:
            masks, scores, logits = self.sam.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])

        detections.mask = np.array(result_masks)

        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        labels = [
            f"{classes[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _ 
            in detections]
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        with sv.ImageSink(target_dir_path="annotated_images", image_name_pattern=f"{self.filename}_annotated.png") as sink:
            sink.save_image(image=annotated_image)



        labels_dir = os.path.dirname(image_path.replace("images", "labels"))
        os.makedirs(labels_dir, exist_ok=True)  # This creates the directory if it doesn't exist

        detections_path = os.path.splitext(image_path.replace("images", "labels"))[0] + ".txt"

        with open(detections_path, 'w') as f:
            i = 0
            for detection in detections.xyxy:
                x_center, y_center, width, height = self.convert_to_yolo_format(image, detection)
                class_id = detections.class_id[i]  # Assuming the class ID is at the 6th position
                i += 1
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

        return image, detections


    def convert_to_yolo_format(self, image, detection):
        dw = 1. / image.shape[1]
        dh = 1. / image.shape[0]
        x = (detection[0] + detection[2]) / 2.0
        y = (detection[1] + detection[3]) / 2.0
        w = detection[2] - detection[0]
        h = detection[3] - detection[1]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return x, y, w, h


x = dino()
x.load_dino_sam()


classes = ['cars']
x.annotate_images("images", "labels", classes)