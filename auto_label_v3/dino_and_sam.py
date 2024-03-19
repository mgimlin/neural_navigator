import os
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
import torch
import cv2
import supervision as sv

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
        self.sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)


    def annotate(self, image_path):
        CLASSES = ['car', 'dog', 'person', 'nose', 'chair', 'shoe', 'ear']
        BOX_TRESHOLD = 0.35
        TEXT_TRESHOLD = 0.25

        image = cv2.imread(image_path)

        # detect objects
        detections = self.model.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )

        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        labels = [
            f"{CLASSES[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _ 
            in detections]
        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

        sv.plot_image(annotated_frame, (16, 16))

