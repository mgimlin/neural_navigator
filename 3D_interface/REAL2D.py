from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from ultralytics import YOLO
import cv2
import torch
import pywavefront


model = YOLO('best.pt')
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    exit()

model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# Window dimensions
WIDTH = 800
HEIGHT = 600

person = pywavefront.Wavefront('objects/BaseMesh.obj', collect_faces=True)
car = pywavefront.Wavefront('objects/car.obj', collect_faces=True)
cow = pywavefront.Wavefront('objects/cow.obj', collect_faces=True)
bus = pywavefront.Wavefront('objects/bus.obj', collect_faces=True)
dog = pywavefront.Wavefront('objects/dog.obj', collect_faces=True)
motorcycle = pywavefront.Wavefront('objects/motorcycle.obj', collect_faces=True)
stpsgn = pywavefront.Wavefront('objects/StopSign.obj', collect_faces=True)
bike = pywavefront.Wavefront('objects/bike.obj', collect_faces=True)
truck = pywavefront.Wavefront('objects/truck.obj', collect_faces=True)
cone = pywavefront.Wavefront('objects/cone.obj', collect_faces=True)
cat = pywavefront.Wavefront('objects/cat.obj', collect_faces=True)
horse = pywavefront.Wavefront('objects/Horse.obj', collect_faces=True)
scooter = pywavefront.Wavefront('objects/scooter.obj', collect_faces=True)
trafficLight = pywavefront.Wavefront('objects/trafficLight.obj', collect_faces=True)
ball = pywavefront.Wavefront('objects/ball.obj', collect_faces=True)
trash = pywavefront.Wavefront('objects/trash.obj', collect_faces=True)
sheep = pywavefront.Wavefront('objects/sheep.obj', collect_faces=True)
skateboard = pywavefront.Wavefront('objects/skateboard.obj', collect_faces=True)
frisbee = pywavefront.Wavefront('objects/frisbee.obj', collect_faces=True)

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

import math

AVERAGE_HEIGHTS = {
    'person': 1.77,
}

CAMERA_HEIGHT = 1
CAMERA_FOV_X = 90
CAMERA_FOV_Y = 50
CAMERA_TILT = 0

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
    
    
classDict = {
    0: person,        
    1: bike,       
    2: car,
    3: motorcycle,   
    4: bus,
    5: truck,
    6: stpsgn,    
    7: cat,
    8: dog,
    9: horse,
    10: sheep,
    11: cow,
    12: frisbee,
    13: ball,
    14: skateboard,
    15: "GTL", 
    16: "RTL",
    17: "YTL",
    18: cube,
    19: scooter,
    20: truck,
    21: cone,
    22: trash,
    23: car
    }


def Model(obj):
    # glPushMatrix()
    # glScalef(*scene_scale)
    # glTranslatef(*scene_trans)

    glColor3f(0.75, 0.75, 0.75)  # Set the color to red

    for mesh in obj.mesh_list:
        glBegin(GL_TRIANGLES)
        for face in mesh.faces:
            for vertex_i in face:
                glVertex3f(*obj.vertices[vertex_i])
        glEnd()
        
        
def display() -> None:
    """
    """
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluPerspective(90, WIDTH / HEIGHT, 0.1, 50.0)
    glTranslatef(0.0, -0.5, -5)
    glRotatef(45, 1, 0, 0)

    # Get a frame from the camera.
    ret, frame = cam.read()
    if not ret:
        return

    # Object detection.
    results = model(frame)

    for result in results:
        for i,box in enumerate(result.boxes.xyxyn):
            # Estimate the distance from the camera to the object.
            # depth = estimate_depth(box[1], box[3])
            depth = -4 * absolute_depth(box[1], box[3])
            x = calculate_x(box[0], box[2], depth)
            y = min(10 + depth, 5)
            cls = (result.boxes.cls)[i]


            glPushMatrix()
            glScalef(0.25, 0.25, 0.25)
            glTranslatef(x, -0.5, y)
            Model(classDict[int(cls.item())])

            modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
            projection = glGetDoublev(GL_PROJECTION_MATRIX)
            viewport = glGetIntegerv(GL_VIEWPORT)
            x_2d, y_2d, z_2d = gluProject(0, 1, 0, modelview, projection, viewport)

            abs_depth = absolute_depth(box[1], box[3])
            draw_text(x_2d, y_2d, f'{abs_depth:.2f}m')
            
            glPopMatrix()
            
    glutSwapBuffers()

def update(value: int) -> None:
    """
    """
    glutPostRedisplay()
    glutTimerFunc(1000 // 60, update, 0)

def main() -> None:
    """
    """
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
    main()
