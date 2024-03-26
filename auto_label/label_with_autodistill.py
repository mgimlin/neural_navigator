import os
import shutil
from PIL import Image
from tqdm import tqdm  # Import tqdm
import supervision as sv
import cv2
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM
import roboflow
from roboflow import Roboflow




def remove_dir(directory):
    if os.path.isdir(directory):
        shutil.rmtree(directory)
        print(f"Directory {directory} and its contents removed successfully.")
    else:
        print(f"Directory {directory} does not exist.")



class label_with_autodistill:
    def __init__(self, home_dir=os.getcwd()):
        self.home_dir = home_dir
        self.images_dir = os.path.join(self.home_dir, "images")
        self.annotated_images_dir = os.path.join(self.home_dir, "annotated_images")
        self.dataset_dir = os.path.join(self.home_dir, "dataset")
        self.run_dir = os.path.join(self.home_dir, "runs")


    def remove_folders(self):
        print("Removing folders...")
        remove_dir(self.images_dir)
        remove_dir(self.annotated_images_dir)
        remove_dir(self.dataset_dir)
        remove_dir(self.run_dir)
        

    def convert_images_to_png(self, input_images_folder):
        try:
            os.mkdir(self.images_dir)
            print(f"Directory {self.images_dir} created successfully.")
        except FileExistsError:
            print(f"Directory {self.images_dir} already exists.")
        except Exception as e:
            print(f"Error creating directory: {e}")
        
        # Iterate over all files in the source folder
        print("Converting images to PNG...")
        for filename in tqdm(os.listdir(input_images_folder)):
            if not filename.startswith('.'):  # Ignore hidden files
                # Construct the full file path
                file_path = os.path.join(input_images_folder, filename)
                # Attempt to open the image file
                try:
                    with Image.open(file_path) as img:
                        # Construct the output file path
                        base_filename, _ = os.path.splitext(filename)
                        output_file = os.path.join(self.images_dir, f"{base_filename}.png")
                        # Convert and save the image in .png format
                        img.save(output_file, "PNG")
                except IOError:
                    print(f"Failed to convert {filename}. It might not be an image file.")


    def annotated_and_save_images(self):
        images_directory_path = os.path.join(self.dataset_dir, "train", "images")
        annotations_directory_path = os.path.join(self.dataset_dir, "train", "labels")
        data_yaml_path = os.path.join(self.dataset_dir, "train", "data.yaml")

        dataset = sv.DetectionDataset.from_yolo(
            images_directory_path=images_directory_path,
            annotations_directory_path=annotations_directory_path,
            data_yaml_path=data_yaml_path)

        len(dataset)

        image_names = list(dataset.images.keys())[:20]

        mask_annotator = sv.MaskAnnotator()
        box_annotator = sv.BoxAnnotator()

        print(dataset.classes)

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

            output_path = os.path.join(output_directory, f"{image_name}_annotated.png")

            with sv.ImageSink(target_dir_path=output_path) as sink:
                sink.save_image(image=annotates_image)


    def train(self, ontology):
        ontology=CaptionOntology(ontology=ontology)
        base_model = GroundedSAM(ontology=ontology)
        self.dataset = base_model.label(
            input_folder=self.images_dir,
            extension=".png",
            output_folder=self.dataset_dir)
        

    def upload_annotations(self):
        # # Login to Roboflow
        roboflow.login()

        # # Access your workspace
        workspace = Roboflow().workspace()

        # # Assuming `dataset.name` is available from the previous steps
        # # Replace 'your_dataset_name_here' with the actual name of your dataset
        dataset_name = 'upload-test'

        # # Create a new project in Roboflow
        new_project = workspace.create_project(
            project_name=dataset_name,
            project_license="MIT",
            project_type="instance-segmentation",
            annotation=f"{dataset_name}-yolo-format")
        
        # rf = roboflow.Roboflow()
        # rf.workspace(workspace).project(project)

        # # Directory paths
        images_dir = os.path.join(self.dataset_dir, "train", "images")
        annotations_dir = os.path.join(self.dataset_dir, "train", "labels")

        # List all image files
        image_paths = sv.list_files_with_extensions(directory=images_dir, extensions=["jpg", "jpeg", "png"])

        # # Upload images and their annotations to Roboflow
        for image_path in tqdm(image_paths):
            annotation_name = f"{image_path.stem}.txt"  # YOLO annotations are .txt files
            annotation_path = os.path.join(annotations_dir, annotation_name)
            
            # Only upload if the annotation file exists for the image
            if os.path.exists(annotation_path):
                new_project.upload(
                    image_path=str(image_path),
                    annotation_path=annotation_path,
                    split="train",  # Assuming all images are for training
                    is_prediction=False,  # These are ground-truth annotations
                    overwrite=True,  # Overwrite existing files if necessary
                    tag_names=["auto-annotated-with-autodistill"],
                    batch_name="auto-annotated-batch"
                )


x = label_with_autodistill()

# x.remove_folders()
# print()
# input_images_folder = os.path.join(os.getcwd(), "images_input")
# x.convert_images_to_png(input_images_folder)
# print()
x.upload_annotations()
