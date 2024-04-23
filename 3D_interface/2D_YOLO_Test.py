from ultralytics import YOLO
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

import supervision as sv
import cv2
import math
import threading

global x, y, running
x, y = 720.0, 540.0
running = True

def drawCircle(x, y, radius=20):
    segments = 32
    glBegin(GL_TRIANGLE_FAN)
    glVertex2f(x, y)
    for i in range(segments + 1):
        angle = 2 * math.pi * i / segments
        glVertex2f(x + (radius * math.cos(angle)), y + (radius * math.sin(angle)))
    glEnd()

def refresh2d(width, height):
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0.0, width, 0.0, height)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def display(): 
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    refresh2d(1440, 1080)

    glColor3f(0.0, 0.0, 1.0)
    global x, y
    drawCircle(1440 - x, 1080 - y, 10)

    glutSwapBuffers()

def update(value):
    global running
    if not running:
        sys.exit(0)
    glutPostRedisplay()
    glutTimerFunc(1000 // 60, update, 0)

def initOpenGL():
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH)
    glutInitWindowSize(1440, 1080)
    glutInitWindowPosition(0, 0)
    glutCreateWindow("PyOpenGlExample")
    glutDisplayFunc(display)
    glutTimerFunc(1000 // 60, update, 0)
    glutMainLoop()

def yoloProcess():
    model = YOLO("yolov8n-seg.pt")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1440)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    specificClassID = 67 # change for a different class you want the info for

    global x, y, running
    while running:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")

        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)

        specificClass = [detection for detection in detections if detection[3] == specificClassID]
        for detection in specificClass:
            specificClassBB = detection[0]

            x1, y1, x2, y2 = specificClassBB

            xCenter = (x1 + x2) / 2
            yCenter = (y1 + y2) / 2

            x, y = xCenter, yCenter
            print(x, y)

    cap.release()

if __name__ == "__main__":
    try:
        threading.Thread(target=initOpenGL, args=(), daemon=True).start()
        yoloProcess()
    except:
        print("Program terminated by user")
        running = False

