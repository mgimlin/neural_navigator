from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

from ultralytics import YOLO

import cv2
import math
import pywavefront
import threading
import sys

# Window dimensions
WIDTH = 800
HEIGHT = 600

AVERAGE_HEIGHTS = {
    'person': 1.77,
}

CAMERA_HEIGHT = 1
CAMERA_FOV_X = 90
CAMERA_FOV_Y = 50
CAMERA_TILT = 0

running = True
results = None
models = {}

model_files = {
    'person': 'objs/person.obj',
}

# Define vertices of the cube
vertices = (
    (1, -1, -1), (1, 1, -1),
    (-1, 1, -1), (-1, -1, -1),
    (1, -1, 1), (1, 1, 1),
    (-1, -1, 1), (-1, 1, 1)
)

# Define faces of the cube using vertices indices
faces = (
    (0, 1, 2, 3),
    (3, 2, 7, 6),
    (6, 7, 5, 4),
    (4, 5, 1, 0),
    (1, 5, 7, 2),
    (4, 0, 3, 6)
)

# Define colors for each face
colors = (
    (1, 0, 0), (0, 1, 0), (0, 0, 1),
    (1, 1, 0), (1, 0, 1), (0, 1, 1)
)

model = YOLO('../best.pt')
cam = cv2.VideoCapture(1)
if not cam.isOpened():
    exit()
    
# import torch

# model_type = "MiDaS_small"
# midas = torch.hub.load("intel-isl/MiDaS", model_type)
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# midas.to(device)
# midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
# transform = midas_transforms.small_transform

def preload_models():
    global models
    for key, file_path in model_files.items():
        try:
            models[key] = pywavefront.Wavefront(file_path, collect_faces=True)
        except Exception as e:
            print(f'{e}')
        
def draw_model(model_name):
    if model_name not in models:
        print(f"Model {model_name} not found")
        return
    
    glScalef(0.75, 0.75, 0.75) 
    glRotatef(0, 0, 0)
    glTranslatef(0, -0.5, 0)
    
    model = models[model_name]
    
    glColor3f(.75, 0.75, 0.75)
    
    glBegin(GL_TRIANGLES)
    for mesh in model.mesh_list:
        for face in mesh.faces:
            for vertex_index in face:
                glVertex3f(*model.vertices[vertex_index])        
        glEnd()
    
def calculate_x(x1: float, x2: float, depth: float) -> float:
    center_x = (x1 + x2) / 2
    angle = CAMERA_FOV_X * (0.5 - center_x)
    x = depth * math.tan(math.radians(angle))

    return x

def absolute_depth(y1: float, y2: float) -> float:
    y_top = min(y1, y2)
    y_bottom = max(y1, y2)
    depth = 0.0

    if y_bottom > 0.99 and y_top < 0.01:
        depth = 1.0

    elif y_bottom > 0.99 and y_top > 0.0:
        opposite = AVERAGE_HEIGHTS['person'] - CAMERA_HEIGHT
        angle = CAMERA_FOV_Y * abs(0.5 - y_top)
        depth = opposite / math.tan(math.radians(angle))

    else:
        angle = CAMERA_FOV_Y * abs(0.5 - y_bottom)
        depth = CAMERA_HEIGHT / math.tan(math.radians(angle))
        depth = 28438.52 * depth**0.00010355 - 28439.64

    return depth

def estimate_depth(y1: float, y2: float) -> float:
    """Estimates depth given the height of a bounding box and the camera parameters.

    Args:
        y1: A normalized y value of the bounding box.
        y2: A normalized y value of the bounding box.

    Returns:
        The estimated distance from the camera to the bounding box.
    """
    y_top = min(y1, y2)
    y_bottom = max(y1, y2)
    depth = 0.0

    if y_bottom > 0.99 and y_top < 0.01:
        depth = 10.0

    elif y_bottom > 0.99 and y_top > 0.0:
        opposite = AVERAGE_HEIGHTS['person'] - CAMERA_HEIGHT
        angle = CAMERA_FOV_Y * abs(0.5 - y_top)
        depth = opposite / math.tan(math.radians(angle))
        # depth *= -depth
        depth /= 2

    else:
        angle = CAMERA_FOV_Y * abs(0.5 - y_bottom)
        depth = CAMERA_HEIGHT / math.tan(math.radians(angle))
        # depth *= -(depth / 4)
        depth *= -1

    return depth

def draw_text(x: float, y: float, text: str) -> None:
    glColor3fv((0, 0, 0))
    glWindowPos2f(x, y)
    for char in text:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))

def cube() -> None:
    """Adds a cube to the scene.
    """
    glBegin(GL_QUADS)
    for i, face in enumerate(faces):
        glColor3fv(colors[i])  # Set color for each face
        for vertex in face:
            glVertex3fv(vertices[vertex])
    glEnd()

def display() -> None:
    """
    """
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluPerspective(90, WIDTH / HEIGHT, 0.1, 50.0)
    glTranslatef(0.0, -0.5, -5)
    glRotatef(45, 1, 0, 0)

    global results
    
    # if not results:
    #     glutSwapBuffers()
    #     return

    # for result in results:
    #     for box in result.boxes.xyxyn:
    #         # Estimate the distance from the camera to the object.
    #         # depth = estimate_depth(box[1], box[3])
    #         # depth = -4 * absolute_depth(box[1], box[3])
    #         # x = calculate_x(box[0], box[2], depth)
    #         # y = min(10 + depth, 5)

    #         glPushMatrix()
    #         glScalef(0.25, 0.25, 0.25)
    #         glTranslatef(10, -0.5, 10)
    #         cube()

    #         modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
    #         projection = glGetDoublev(GL_PROJECTION_MATRIX)
    #         viewport = glGetIntegerv(GL_VIEWPORT)
    #         x_2d, y_2d, z_2d = gluProject(0, 1, 0, modelview, projection, viewport)

    #         # abs_depth = absolute_depth(box[1], box[3])
    #         # draw_text(x_2d, y_2d, f'{abs_depth:.2f}m')
            
    #         glPopMatrix()
    
    # print("working")
    
    glPushMatrix()
    draw_model('person')
    
    modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
    projection = glGetDoublev(GL_PROJECTION_MATRIX)
    viewport = glGetIntegerv(GL_VIEWPORT)
    gluProject(0, 1, 0, modelview, projection, viewport)
    glPopMatrix()
            
    glutSwapBuffers()

def update(value: int) -> None:
    """
    """
    glutPostRedisplay()
    glutTimerFunc(1000 // 60, update, 0)
    
def yolo_thread() -> None:
    global results
    global running
    
    print('starting yolo thread')
    while running:
        ret, frame = cam.read()
        if not ret:
            return
        
        results = model(frame)

def main() -> None:
    """
    """
    preload_models()
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(WIDTH, HEIGHT)
    glutCreateWindow("3D Interface")
    glutDisplayFunc(display)
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glutTimerFunc(1000 // 60, update, 0)
    glEnable(GL_DEPTH_TEST)
    glutMainLoop()

if __name__ == "__main__":
    try:
        # threading.Thread(target=yolo_thread, args=(), daemon=True).start()
        main()
    except:
        running = False
        sys.exit()
