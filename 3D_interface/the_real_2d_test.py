from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from ultralytics import YOLO
import cv2
import math
import pywavefront
import threading
import sys
import time

model = None # YOLOv8 global.
cam = None # Webcam global.

running = True # The flag to stop the YOLOv8 thread.
results = None # YOLOv8 results global.

deltaZ = 0.0 # ?

# obj file stuff?
models = {}
model_files = {
    'person': 'objs/person.obj',
}

# Globals for the depth estimation.
AVERAGE_HEIGHTS = {
    'person': 1.77,
}
CAMERA_HEIGHT = 1
CAMERA_FOV_X = 90
CAMERA_FOV_Y = 50
CAMERA_TILT = 0

<<<<<<< HEAD
# Globals for the window dimensions
WIDTH = 800
HEIGHT = 600
=======
deltaZ = 0.0

running = True
results = None
models = {}

# classDict = {
#     0: person,        
#     1: bike,       
#     2: car,
#     3: motorcycle,   
#     4: bus,
#     5: truck,
#     6: stpsgn,    
#     7: cat,
#     8: dog,
#     9: horse,
#     10: sheep,
#     11: cow,
#     12: frisbee,
#     13: ball,
#     14: skateboard,
#     15: trafficLight, 
#     16: trafficLight,
#     17: trafficLight,
#     18: cube(),
#     19: scooter,
#     20: truck,
#     21: cone,
#     22: trash,
#     23: car
# } #15=gtl 16=rtl, 17=ytl
model_files = {
    # 'person': 'objs/person.obj',
    # 'bike': 'objs/bike.obj',
    # 'car': 'objs/car.obj',
    # 'motorcycle': 'objs/motorcycle.obj',
    # 'truck': 'objs/truck.obj',
    # 'stopSign': 'objs/stopSign.obj',
    # 'cat': 'objs/cat.obj',
    'dog': 'objs/dog.obj',
    # 'horse': 'objs/horse.obj',
    # 'sheep': 'objs/sheep.obj',
    # 'cow': 'objs/cow.obj',
    # 'frisbee': 'objs/frisbee.obj',
    # 'ball': 'objs/ball.obj',
    # 'skateboard': 'objs/skateboard.obj',
    # 'trafficLight': 'objs/trafficLight.obj',
    # 'scooter': 'objs/scooter.obj',
    # 'truck': 'objs/truck.obj',
    # 'cone': 'objs/cone.obj',
    # 'trash': 'objs/trash.obj'
}
>>>>>>> 124436b8d39ebb870218a2f2f9a3bb088b3b01bd

# Globals used for rending a cube.
VERTICES = (
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, -1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, -1, 1),
    (-1, 1, 1),
)
FACES = (
    (0, 1, 2, 3),
    (3, 2, 7, 6),
    (6, 7, 5, 4),
    (4, 5, 1, 0),
    (1, 5, 7, 2),
    (4, 0, 3, 6),
)
COLORS = (
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
)

<<<<<<< HEAD
def preload_models() -> None:
    """
    ?
    """
=======
# model = YOLO('../best.pt')
# cam = cv2.VideoCapture(1)
# if not cam.isOpened():
#     exit()
    
# import torch

# model_type = "MiDaS_small"
# midas = torch.hub.load("intel-isl/MiDaS", model_type)
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# midas.to(device)
# midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
# transform = midas_transforms.small_transform

def preload_models():
>>>>>>> 124436b8d39ebb870218a2f2f9a3bb088b3b01bd
    global models
    for key, file_path in model_files.items():
        try:
            models[key] = pywavefront.Wavefront(file_path, collect_FACES=True)
        except Exception as e:
            print(f'{e}')
        
def draw_model(model_name: str) -> None:
    """
    ?
    """
    global deltaZ
    if model_name not in models:
        print(f"Model {model_name} not found")
        return
    
    glScalef(.02, .02, .02) 
    # glRotatef(180, 1, 0, 0)
    # glRotatef(90, 0, 1,  0)
    # glRotatef(-20, 0, 0,  1)
    glTranslatef(0, -0.5, 2 - deltaZ)
    
    model = models[model_name]
    
    glColor3f(.75, 0.75, 0.75)
    
    for mesh in model.mesh_list:
<<<<<<< HEAD
        for face in mesh.FACES:
=======
        glBegin(GL_TRIANGLES)
        for face in mesh.faces:
>>>>>>> 124436b8d39ebb870218a2f2f9a3bb088b3b01bd
            for vertex_index in face:
                glVertex3f(*model.VERTICES[vertex_index])        
        glEnd()
    
def estimate_x(x1: float, x2: float, depth: float) -> float:
    """Estimates the x value for the UI position.

    Args:
        x1: A normalized x value of the bounding box.
        x2: A normalized x value of the bounding box.
        depth: The exaggerated UI depth estimate.

    Returns:
        The x value for the exaggerated UI position.
    """
    center_x = (x1 + x2) / 2
    angle = CAMERA_FOV_X * (0.5 - center_x)
    x = depth * math.tan(math.radians(angle))

    return x

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

    # The object is covering the entire frame.
    if y_bottom > 0.99 and y_top < 0.01:
        return 1.0

    # The object is cut off by the bottom of the frame.
    elif y_bottom > 0.99 and y_top > 0.0:
        opposite = AVERAGE_HEIGHTS['person'] - CAMERA_HEIGHT
        angle = CAMERA_FOV_Y * abs(0.5 - y_top)
        return opposite / math.tan(math.radians(angle))

    # The object is fully in the frame.
    else:
        angle = CAMERA_FOV_Y * abs(0.5 - y_bottom)
        depth = CAMERA_HEIGHT / math.tan(math.radians(angle))
        return 28438.52 * depth**0.00010355 - 28439.64

def draw_text(x: float, y: float, text: str) -> None:
    """Renders text on the screen.

    Args:
        x: The window x value for the text.
        y: The window y value for the text.
        text: A message to render.
    """
    glColor3fv((0, 0, 0))
    glWindowPos2f(x, y)
    for char in text:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))

def cube() -> None:
    """Adds a cube to the scene.
    """
    glBegin(GL_QUADS)
    for i, face in enumerate(FACES):
        glColor3fv(COLORS[i])
        for vertex in face:
            glVertex3fv(VERTICES[vertex])
    glEnd()

def display() -> None:
    """Pulls the most recent results from the YOLOv8 thread, and then renders
    the environment.
    """
    global results
    
    if not results:
        glutSwapBuffers()
        return
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluPerspective(90, WIDTH / HEIGHT, 0.1, 50.0)
    glTranslatef(0.0, -0.5, -5)
    glRotatef(45, 1, 0, 0)

    for result in results:
        for box in result.boxes.xyxyn:
            abs_depth = estimate_depth(box[1], box[3]) # The actual estimated depth.
            ui_depth = -4 * abs_depth # The exaggerated depth for the UI.
            
            # Place the object in the environment.
            x = estimate_x(box[0], box[2], ui_depth)
            y = min(10 + ui_depth, 5)
            glPushMatrix()
            glScalef(0.25, 0.25, 0.25)
            glTranslatef(x, -0.5, y)
            cube()

            # Display text above the object.
            modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
            projection = glGetDoublev(GL_PROJECTION_MATRIX)
            viewport = glGetIntegerv(GL_VIEWPORT)
            x_2d, y_2d, z_2d = gluProject(0, 1, 0, modelview, projection, viewport)
            draw_text(x_2d, y_2d, f'{abs_depth:.2f}m')
            
            glPopMatrix()
    
    # FIXME
    # what is this
    
    # glPushMatrix()
    # draw_model('person')
    # 
    # modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
    # projection = glGetDoublev(GL_PROJECTION_MATRIX)
    # viewport = glGetIntegerv(GL_VIEWPORT)
    # gluProject(0, 1, 0, modelview, projection, viewport)
    # glPopMatrix()
            
    glutSwapBuffers()

def update(value: int) -> None:
    """Refreshes the window.

    Args:
        value: idk
    """
    glutPostRedisplay()
    glutTimerFunc(1000 // 60, update, 0)
    
def yolo_thread() -> None:
    """Continuously makes detections with YOLOv8. Updates the ``results``
    global.
    """
    global results
    global running
    global model
    global cam
    
    print('starting yolo thread')
    while running:
        if not model or not cam:
            continue
        
        ret, frame = cam.read()
        if not ret:
            return
        
        results = model(frame)

def move_backwards() -> None:
    """
    ?
    """
    global deltaZ
    
    for i in range(40):
        deltaZ += 1
        glutPostRedisplay()
        time.sleep(0.5)
        

def main() -> None:
    """Sets up YOLOv8 and the webcam. Then, launches the YOLOv8 thread and
    starts the display loop.
    """
    global model
    global cam
    global running

    # Set up YOLOv8.
    model = YOLO('../best.pt')

    # Set up the webcam.
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        return
    
    # Launch the YOLOv8 thread.
    threading.Thread(target=yolo_thread, args=(), daemon=True).start()
    
    preload_models()
    
    # Setup OpenGL and run the display loop.
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(WIDTH, HEIGHT)
    glutCreateWindow("3D Interface")
    glutDisplayFunc(display)
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glutTimerFunc(1000 // 60, update, 0)
    glEnable(GL_DEPTH_TEST)
    glutMainLoop()
    running = False

if __name__ == "__main__":
    try:
        main()
    except:
        running = False
        sys.exit()
