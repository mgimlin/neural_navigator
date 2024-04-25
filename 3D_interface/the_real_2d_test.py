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

last_depths = {}
current_depths = {}

deltaZ = 0.0 # ?

# obj file stuff?
models = {}
model_files = {
    'person': 'objs/person.obj',
}

# Globals constants for the depth estimation.
AVERAGE_HEIGHTS = {
    'person': 1.77,
}
CAMERA_HEIGHT = 1
CAMERA_FOV_X = 90
CAMERA_FOV_Y = 50
CAMERA_TILT = 0

# Globals for the window dimensions
WIDTH = 800
HEIGHT = 600

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

def preload_models() -> None:
    """
    ?
    """
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
        for face in mesh.FACES:
            for vertex_index in face:
                glVertex3f(*model.VERTICES[vertex_index])        
        glEnd()
    
def estimate_x(x1: float, x2: float, depth: float) -> float:
    """Estimates the x value of a position.

    Args:
        x1: A normalized x value of the bounding box.
        x2: A normalized x value of the bounding box.
        depth: An estimated depth.

    Returns:
        The x value of the position.
    """
    center_x = (x1 + x2) / 2
    angle = CAMERA_FOV_X * (0.5 - center_x)
    return depth * math.tan(math.radians(angle))

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
        
        return depth
        # return 28438.52 * depth**0.00010355 - 28439.64

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
    global last_depths
    global current_depths
    
    if not results:
        glutSwapBuffers()
        return
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluPerspective(90, WIDTH / HEIGHT, 0.1, 50.0)
    glTranslatef(0.0, -0.5, -5)
    glRotatef(45, 1, 0, 0)

    for result in results:
        # boxes = result.boxes.xyxyn
        # track_ids = result.boxes.id.int().cpu().tolist()
        # for box, track_id in zip(boxes, track_ids):
        for box in result.boxes.xyxyn:
            abs_depth = estimate_depth(box[1], box[3]) # The actual estimated depth.
            abs_x = estimate_x(box[0], box[2], abs_depth)
            abs_y = abs_depth

            ui_depth = -4 * abs_depth # The exaggerated depth for the UI.
            ui_x = estimate_x(box[0], box[2], ui_depth)
            ui_y = min(10 + ui_depth, 5)

            # current_depths[track_id] = abs_depth
            
            # Place the object in the environment.
            glPushMatrix()
            glScalef(0.25, 0.25, 0.25)
            glTranslatef(ui_x, -0.5, ui_y)
            cube()

            # Display text above the object.
            modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
            projection = glGetDoublev(GL_PROJECTION_MATRIX)
            viewport = glGetIntegerv(GL_VIEWPORT)
            x_2d, y_2d, z_2d = gluProject(0, 1, 0, modelview, projection, viewport)
            draw_text(x_2d, y_2d, f'{actual_depth:.2f}m')
            
            glPopMatrix()

            # last_depths[track_id] = current_depths[track_id]
    
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
        
        results = model.track(frame)

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
    cam = cv2.VideoCapture(1)
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
