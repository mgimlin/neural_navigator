from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

import sys

import models

# Globals for the window dimensions
WIDTH = 800
HEIGHT = 600

def display() -> None:
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glScale(0.25, 0.25, 0.25)
    glTranslatef(0.0, -0.5, 0)
    glRotatef(45, 1, 0, 0)
    
    glPushMatrix()
    models.draw_model(0)
    
    modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
    projection = glGetDoublev(GL_PROJECTION_MATRIX)
    viewport = glGetIntegerv(GL_VIEWPORT)
    gluProject(0, 1, 0, modelview, projection, viewport)
    glPopMatrix()
            
    glutSwapBuffers()

def update(value: int) -> None:
    """Refreshes the window.

    Args:
        value: idk
    """
    glutPostRedisplay()
    glutTimerFunc(1000 // 60, update, 0)

def main() -> None:
    # Setup OpenGL and run the display loop.
    models.preload_models()
    
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(WIDTH, HEIGHT)
    glutCreateWindow("3D Interface")
    glutDisplayFunc(display)
    glutTimerFunc(1000 // 60, update, 0)
    gluPerspective(90, WIDTH / HEIGHT, 0.1, 50.0)
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glEnable(GL_DEPTH_TEST)
    glutMainLoop()

if __name__ == "__main__":
    try:
        main()
    except:
        print("Program terminated")
        sys.exit()