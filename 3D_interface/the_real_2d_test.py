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
import models

model = None # YOLOv8 global.
cam = None # Webcam global.

running = True # The flag to stop the YOLOv8 thread.
results = None # YOLOv8 results global.

last_time = 0.0
last_positions = {}
new_frame = False
velocities = {}

deltaZ = 0.0 # ?

# obj file stuff?
# models = {}
model_files = {
    'person': 'objs/person.obj',
}

# Globals constants for the depth estimation.
AVERAGE_HEIGHTS = {
    0: 1.77,
    1: 1.05,
    2: 1.8,
    3: 1.05,
    4: 3.9624,
    5: 3.9624,
    6: 2.1336,
    7:  0.5,
    8:  0.6,
    9:  2.1,
    10: 1.0,
    11: 1.0,
    12: 0.05,
    13: 0.1,
    14: 0.1,
    15: 1.2192,
    16: 1.2192,
    17: 1.2192,
    18: 0.9,
    19: 1.05,
    20: 3.9624,
    21: 0.28,
    22: 1.1176,
    23: 1.1176,

}
CAMERA_HEIGHT = 0.99
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

# def preload_models() -> None:
#     """
#     ?
#     """
#     global models
#     for key, file_path in model_files.items():
#         try:
#             models[key] = pywavefront.Wavefront(file_path, collect_FACES=True)
#         except Exception as e:
#             print(f'{e}')
        
# def draw_model(model_name: str) -> None:
#     """
#     ?
#     """
#     global deltaZ
#     if model_name not in models:
#         print(f"Model {model_name} not found")
#         return
    
#     glScalef(.02, .02, .02) 
#     # glRotatef(180, 1, 0, 0)
#     # glRotatef(90, 0, 1,  0)
#     # glRotatef(-20, 0, 0,  1)
#     glTranslatef(0, -0.5, 2 - deltaZ)
    
#     model = models[model_name]
    
#     glColor3f(.75, 0.75, 0.75)
    
#     for mesh in model.mesh_list:
#         for face in mesh.FACES:
#             for vertex_index in face:
#                 glVertex3f(*model.VERTICES[vertex_index])        
#         glEnd()
    
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

def estimate_depth(cls: int, y1: float, y2: float) -> float:
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
        opposite = AVERAGE_HEIGHTS[cls] - CAMERA_HEIGHT
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
    global last_time
    global last_positions
    global new_frame
    global velocities

    current_time = time.time()
    
    if not results:
        glutSwapBuffers()
        return
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluPerspective(90, WIDTH / HEIGHT, 0.1, 50.0)
    glTranslatef(0.0, -0.5, -5)
    glRotatef(45, 1, 0, 0)

    for result in results:
        boxes = result.boxes.xyxyn
        if result.boxes.id == None:
            break
        track_ids = result.boxes.id.int().cpu().tolist()
        classes = result.boxes.cls
        for box, track_id, cls in zip(boxes, track_ids, classes):
            cls = int(cls.item())
            abs_depth = estimate_depth(cls, box[1], box[3]) # The actual estimated depth.
            abs_x = estimate_x(box[0], box[2], abs_depth)
            abs_y = abs_depth

            ui_depth = -4 * abs_depth # The exaggerated depth for the UI.
            ui_x = estimate_x(box[0], box[2], ui_depth)
            ui_y = min(10 + ui_depth, 5)

            if not track_id in velocities:
                velocities[track_id] = [0.0, 0.0, 0.0, 0.0]

            if not track_id in velocities or new_frame:
                velocity = 0.0
                if track_id in last_positions:
                    vel_x = abs(abs_x - last_positions[track_id][0]) / \
                        (current_time - last_time)
                    vel_y = abs(abs_y - last_positions[track_id][1]) / \
                        (current_time - last_time)
                    velocity = math.sqrt(vel_x**2 + vel_y**2)

                velocities[track_id].append(velocity)
                if len(velocities) > 4:
                    velocities.pop(0)

            smooth_vel = \
                0.25 * velocities[track_id][-1] + \
                0.25 * velocities[track_id][-2] + \
                0.25 * velocities[track_id][-3] + \
                0.25 * velocities[track_id][-4]

            # Place the object in the environment.
            glPushMatrix()
            models.draw_model(cls, ui_x, ui_y)
            # glScalef(0.25, 0.25, 0.25)
            # glTranslatef(ui_x, -0.5, ui_y)
            # cube()

            # Display text above the object.
            modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
            projection = glGetDoublev(GL_PROJECTION_MATRIX)
            viewport = glGetIntegerv(GL_VIEWPORT)
            x_2d, y_2d, z_2d = gluProject(0, 1, 0, modelview, projection, viewport)
            draw_text(x_2d, y_2d, f'{abs_depth:.2f}m {smooth_vel:.2f} m/s')
            print("here")
            glPopMatrix()

            if not track_id in last_positions or new_frame:
                last_positions[track_id] = (abs_x, abs_y)

    last_time = current_time
    new_frame = False
    
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
    global new_frame
    
    print('starting yolo thread')
    while running:
        if not model or not cam:
            continue
        
        ret, frame = cam.read()
        if not ret:
            return
        
        results = model.track(frame)
        new_frame = True

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
    models.preload_models()


    # Set up YOLOv8.
    model = YOLO('../best.pt')

    # Set up the webcam.
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        return
    
    # Launch the YOLOv8 thread.
    threading.Thread(target=yolo_thread, args=(), daemon=True).start()
    
    # preload_models()
    
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
    except Exception as e:
        print(e)
        running = False
        sys.exit()
