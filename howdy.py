from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

from ultralytics import YOLO
import cv2
import torch
import numpy as np
import pywavefront

# Import Monodepth model
from monodepth2 import monodepth2
# Initialize YOLO model
model = YOLO('best.pt')

# Initialize Monodepth model
monodepth_model = monodepth2()

# Camera initialization
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    exit()

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

def cube():
    glBegin(GL_QUADS)
    for i, face in enumerate(faces):
        glColor3fv(colors[i])  # Set color for each face
        for vertex in face:
            glVertex3fv(vertices[vertex])
    glEnd()

obj = pywavefront.Wavefront('car.obj')

def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluPerspective(90, (WIDTH / HEIGHT), 0.1, 50.0)
    glTranslatef(0.0, -0.5, -5)
    glRotatef(45, 1, 0, 0)

    ret, frame = cam.read()
    if not ret:
        return

    # Object detection
    results = model(frame)

    # Depth estimation
    depth_map = monodepth_model.eval(frame)

    for result in results:
        for box in result.boxes.xyxy:
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2

            # Get depth at the center of the detected object
            depth = depth_map[int(center_y), int(center_x)]

            glPushMatrix()
            glTranslatef(
                20 * center_x - 10,
                -0.5,
                5 - 20 * depth / 1000
            )
            # Render the loaded object
            pywavefront.visualization.draw(obj)
            glPopMatrix()

    glutSwapBuffers()
# def display():
#     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
#     glLoadIdentity()
#     gluPerspective(90, (WIDTH / HEIGHT), 0.1, 50.0)
#     glTranslatef(0.0, -0.5, -5)
#     glRotatef(45, 1, 0, 0)

#     ret, frame = cam.read()
#     if not ret:
#         return

#     # Object detection
#     results = model(frame)

#     # Depth estimation
#     depth_map = monodepth_model.eval(frame)

#     for result in results:
#         for box in result.boxes.xyxy:
#             center_x = (box[0] + box[2]) / 2
#             center_y = (box[1] + box[3]) / 2

#             # Get depth at the center of the detected object
#             depth = depth_map[int(center_y), int(center_x)]

#             glPushMatrix()
#             glTranslatef(
#                 20 * center_x - 10,
#                 -0.5,
#                 5 - 20 * depth / 1000
#             )
#             cube()
#             glPopMatrix()
#     glutSwapBuffers()

def update(value):
    glutPostRedisplay()
    glutTimerFunc(int(1000 / 60), update, 0)

def main():
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(WIDTH, HEIGHT)
    glutCreateWindow("Object Detection and Depth Estimation")
    glutDisplayFunc(display)
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glutTimerFunc(int(1000 / 60), update, 0)
    glEnable(GL_DEPTH_TEST)
    glutMainLoop()

if __name__ == "__main__":
    main()
# from OpenGL.GL import *
# from OpenGL.GLUT import *
# from OpenGL.GLU import *
# import numpy as np

# # Load object file function
# def load_object(filename):
#     vertices = []
#     faces = []
#     colors = []

#     with open(filename, 'r') as file:
#         for line in file:
#             if line.startswith('v '):
#                 vertex = list(map(float, line[2:].split()))
#                 vertices.append(vertex)
#             elif line.startswith('f '):
#                 face = list(map(int, line[2:].split()))
#                 faces.append([vertex - 1 for vertex in face])
#                 # Assign random color to each face
#                 colors.append(np.random.rand(3))

#     return vertices, faces, colors

# # Load object file
# vertices, faces, colors = load_object('car.obj')

# # Modify rendering function to render loaded object
# def render_object():
#     glBegin(GL_TRIANGLES)
#     for i, face in enumerate(faces):
#         glColor3fv(colors[i])  # Set color for each face
#         for vertex_index in face:
#             glVertex3fv(vertices[vertex_index])
#     glEnd()

# # Modify display function to adjust position and scale of object
# def display():
#     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
#     glLoadIdentity()
#     gluPerspective(90, (WIDTH / HEIGHT), 0.1, 50.0)
#     glTranslatef(0.0, 0.0, -5)  # Adjust position of the object
#     glRotatef(45, 1, 0, 0)

#     ret, frame = cam.read()
#     if not ret:
#         return

#     # Object detection
#     results = model(frame)

#     # Depth estimation
#     depth_map = monodepth_model.eval(frame)

#     for result in results:
#         for box in result.boxes.xyxy:
#             center_x = (box[0] + box[2]) / 2
#             center_y = (box[1] + box[3]) / 2

#             # Get depth at the center of the detected object
#             depth = depth_map[int(center_y), int(center_x)]

#             glPushMatrix()
#             glTranslatef(
#                 20 * center_x - 10,
#                 0.0,  # Adjust height of the object
#                 5 - 20 * depth / 1000
#             )
#             render_object()  # Render loaded object
#             glPopMatrix()

#     glutSwapBuffers()

# # Main function remains the same
# obj = pywavefront.Wavefront('car.obj')

# def display():
#     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
#     glLoadIdentity()
#     gluPerspective(90, (WIDTH / HEIGHT), 0.1, 50.0)
#     glTranslatef(0.0, -0.5, -5)
#     glRotatef(45, 1, 0, 0)

#     ret, frame = cam.read()
#     if not ret:
#         return

#     # Object detection
#     results = model(frame)

#     # Depth estimation
#     depth_map = monodepth_model.eval(frame)

#     for result in results:
#         for box in result.boxes.xyxy:
#             center_x = (box[0] + box[2]) / 2
#             center_y = (box[1] + box[3]) / 2

#             # Get depth at the center of the detected object
#             depth = depth_map[int(center_y), int(center_x)]

#             glPushMatrix()
#             glTranslatef(
#                 20 * center_x - 10,
#                 -0.5,
#                 5 - 20 * depth / 1000
#             )
#             # Render the loaded object
#             pywavefront.visualization.draw(obj)
#             glPopMatrix()

#     glutSwapBuffers()
