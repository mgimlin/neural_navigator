from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import math
import sys

width, height = 500, 400
x, y = 250.0, 200.0

def drawCircle(x, y, radius=20):
    segments = 32
    glBegin(GL_TRIANGLE_FAN)
    glVertex2f(x, y)
    for i in range(segments + 1):
        angle = 2 * math.pi * i / segments
        glVertex2f(x + (radius * math.cos(angle)), y + (radius * math.sin(angle)))
    glEnd()

def refresh2d():
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0.0, width, 0.0, height)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    refresh2d()

    glColor3f(0.0, 0.0, 1.0)
    drawCircle(x, y, 10)

    glutSwapBuffers()

def update(value):
    global x, y
    try:
        x, y = map(float, input("Enter new x y coordinates (e.g., '100.5 200.25'): ").split())
    except ValueError:
        print("Invalid input. Using last known coordinates.")
    glutPostRedisplay()  # Mark the current window as needing to be redisplayed
    glutTimerFunc(1000, update, 0)  # Call 'update' again after 1000 ms

def init():
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH)
    glutInitWindowSize(width, height)
    glutInitWindowPosition(0, 0)
    window = glutCreateWindow("PyOpenGlExample")
    glutDisplayFunc(display)
    glutTimerFunc(1000, update, 0)
    glutMainLoop()

def main():
    init()

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        try:
            x, y = map(float, sys.argv[1:3])
        except ValueError:
            print("Warning: Invalid input. Using default values for x and y")
    else:
        print("No or insuggicient arguments provided. Using default values for x and y.")

    main()

