import os
import shutil
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
import torch
import cv2
import supervision as sv
from tqdm import tqdm
import numpy as np
import roboflow


def remove_dir(directory):
    if os.path.isdir(directory):
        shutil.rmtree(directory)
        print(f"Directory {directory} and its contents removed successfully.")
    else:
        print(f"Directory {directory} does not exist.")


class dino:
    def __init__(self, home_dir=os.getcwd()):
        self.home_dir = home_dir    
        os.makedirs("annotated_images", exist_ok=True)


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


    def annotate_images(self, images_path, classes):
        images = {}
        annotations = {}
        file_names = []

        for filename in tqdm(os.listdir(images_path)):
            if not filename.startswith('.'):  # Ignore hidden files
                file_path = os.path.join(images_path, filename)
                image, detections = self.annotate(file_path, classes)

                images[filename] = image
                annotations[filename] = detections
                file_names.append(filename)

        MIN_IMAGE_AREA_PERCENTAGE = 0.002
        MAX_IMAGE_AREA_PERCENTAGE = 0.80
        APPROXIMATION_PERCENTAGE = 0.75

        sv.DetectionDataset(
            classes=classes,
            images=images,
            annotations=annotations
        ).as_yolo(
            annotations_directory_path='annotations',
            data_yaml_path="data.yaml",
            min_image_area_percentage=MIN_IMAGE_AREA_PERCENTAGE,
            max_image_area_percentage=MAX_IMAGE_AREA_PERCENTAGE,
            approximation_percentage=APPROXIMATION_PERCENTAGE
        )


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

        return image, detections


    def save_images_locally(self):
        images_directory_path = "images"
        annotations_directory_path = "annotations"
        data_yaml_path = "data.yaml"

        dataset = sv.DetectionDataset.from_yolo(
            images_directory_path=images_directory_path,
            annotations_directory_path=annotations_directory_path,
            data_yaml_path=data_yaml_path)

        image_names = list(dataset.images.keys())[:20]

        mask_annotator = sv.MaskAnnotator()
        box_annotator = sv.BoxAnnotator()

        print("classes:", dataset.classes)

        output_directory = os.path.join(self.home_dir, "annotated_images")

        os.makedirs(output_directory, exist_ok=True)

        images = []
        for image_name in image_names:
            image = dataset.images[image_name]
            annotations = dataset.annotations[image_name]
            labels = [
                dataset.classes[class_id]
                for class_id
                in annotations.class_id]
            annotates_image = mask_annotator.annotate(
                scene=image.copy(),
                detections=annotations)
            annotates_image = box_annotator.annotate(
                scene=annotates_image,
                detections=annotations,
                labels=labels)
            images.append(annotates_image)

            output_path = os.path.join(output_directory)

            with sv.ImageSink(target_dir_path=output_path, image_name_pattern=f"{image_name}_annotated.png") as sink:
                sink.save_image(image=annotates_image)


    def upload_annotations(self, project):

        # List all image files
        image_paths = sv.list_files_with_extensions(directory="images", extensions=["jpg", "jpeg", "png"])

        # # Upload images and their annotations to Roboflow
        for image_path in tqdm(image_paths):
            annotation_name = f"{image_path.stem}.txt"  # YOLO annotations are .txt files
            annotation_path = os.path.join("annotations", annotation_name)
            
            # Only upload if the annotation file exists for the image
            if os.path.exists(annotation_path):
                project.upload(
                    image_path=str(image_path),
                    annotation_path=annotation_path,
                    split="train",  # Assuming all images are for training
                    is_prediction=False,  # These are ground-truth annotations
                    overwrite=True,  # Overwrite existing files if necessary
                    tag_names=["auto-annotated-with-autodistill"],
                    batch_name="auto-annotated-batch"
                )



remove_dir("annotations")
remove_dir("annotated_images")


x = dino()
x.load_dino_sam()
classes = ['cars']
x.annotate_images("images", classes)


x.save_images_locally()



workspace = "neuralnavigator-94vew"
dataset_name = 'upload-test-4'

# **** Option 1: open existing project
roboflow.login()
# project = roboflow.Roboflow().workspace(workspace).project(dataset_name)

# **** Option 2: create new project
project = roboflow.Roboflow().workspace(workspace).create_project(
    project_name=dataset_name,
    project_license="MIT",
    project_type="instance-segmentation",
    annotation=f"{dataset_name}-yolo-format")

x.upload_annotations(project)