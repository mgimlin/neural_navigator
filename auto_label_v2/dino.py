import os
import shutil
from PIL import Image
from tqdm import tqdm  # Import tqdm
import supervision as sv
import cv2
import roboflow
from groundingdino.util.inference import load_model, load_image, predict, annotate

# !git clone https://github.com/IDEA-Research/GroundingDINO.git
# %cd {HOME}/GroundingDINO
# !pip install -q -e .
# !pip install -q roboflow

# %cd {HOME}
# !wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth



# !pip install -q 'git+https://github.com/facebookresearch/segment-anything.git'
# !pip install -q jupyter_bbox_widget roboflow dataclasses-json supervision

def remove_dir(directory):
    if os.path.isdir(directory):
        shutil.rmtree(directory)
        print(f"Directory {directory} and its contents removed successfully.")
    else:
        print(f"Directory {directory} does not exist.")



class dino:
    def __init__(self, home_dir=os.getcwd()):
        self.home_dir = home_dir

    def load_dino(self, input_weights_path="groundingdino_swint_ogc.pth", input_config_path="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"):
        config_path = os.path.join(self.home_dir, input_config_path)
        weights_path = os.path.join(self.home_dir, input_weights_path)
        if not (os.path.isfile(config_path) and os.path.isfile(weights_path)):
            print("load_dino: input files do not exist")
            return
        self.model = load_model(config_path, weights_path)

    def draw_boxes(self, image_path):
        TEXT_PROMPT = "car"
        BOX_TRESHOLD = 0.35
        TEXT_TRESHOLD = 0.25

        image_source, image = load_image(image_path)

        boxes, logits, phrases = predict(
            model=self.model, 
            image=image, 
            caption=TEXT_PROMPT, 
            box_threshold=BOX_TRESHOLD, 
            text_threshold=TEXT_TRESHOLD
        )

        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        # %matplotlib inline  
        sv.plot_image(annotated_frame, (16, 16))

    def get_roboflow_images_path(self, workspace, project, dataset):
        roboflow.login()
        rf = roboflow.Roboflow()
        robo_project = rf.workspace(workspace).project(project)
        robo_dataset = robo_project.version(3).download(dataset)
        subdirectory = "all"
        # subdirectory = "valid"

        image_directory_path = f"{robo_dataset.location}/{subdirectory}"
        # image_names = os.listdir(image_directory_path)
        # image_index = randrange(len(image_names))
        # image_name = image_names[0]
        # image_path = os.path.join(image_directory_path, image_name)

        return image_directory_path        
    






