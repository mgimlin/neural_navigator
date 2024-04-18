from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

from ultralytics import YOLO
import cv2

import torch

model = YOLO('../../yolo/best.pt')
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

angle = 0  # Initial angle

def cube():
    glBegin(GL_QUADS)
    for i, face in enumerate(faces):
        glColor3fv(colors[i])  # Set color for each face
        for vertex in face:
            glVertex3fv(vertices[vertex])
    glEnd()

def display():
    global angle
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluPerspective(90, (WIDTH/HEIGHT), 0.1, 50.0)
    glTranslatef(0.0, -0.5, -5)
    glRotatef(45, 1, 0, 0)

    ret, frame = cam.read()
    if not ret:
        return

    # Object detection.
    results = model(frame)
    
    # Depth estimation.
    img = frame
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()

    for result in results:
        for box in result.boxes.xyxyn:
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2

            # print(center_x, len(depth_map[0]), int(center_x * len(depth_map[0])))
            depth = depth_map[int(center_x * len(depth_map))][int(center_y * len(depth_map[0]))]
            
            glPushMatrix()
            glTranslatef(
                20 * center_x - 10,
                -0.5,
                # -5,
                # 0 - 10 * (1 - abs(box[1] - box[3]))
                5 - 20 * depth / 1000
            )
            cube()
            glPopMatrix()

    # Rotate the cube
    # glPushMatrix()
    # glRotatef(angle, 0, 1, 0)  # Rotate around the y-axis
    # cube()
    # glPopMatrix()

    # glPushMatrix()
    # glTranslatef(5, -0.5, -5)
    # cube()
    # glPopMatrix()

    glutSwapBuffers()

    # angle += 1  # Increment the angle for the next frame

def update(value):
    glutPostRedisplay()
    glutTimerFunc(int(1000/60), update, 0)

def main():
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(WIDTH, HEIGHT)
    glutCreateWindow("Rotating Cube")
    glutDisplayFunc(display)
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glutTimerFunc(int(1000/60), update, 0)
    glEnable(GL_DEPTH_TEST)
    glutMainLoop()

if __name__ == "__main__":
    main()
