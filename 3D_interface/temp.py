from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

import sys
import time
import threading

import models

TEMP_BACKWARD_STEPS = 5

DELTA_Z = 0.0

# Globals for the window dimensions
WIDTH = 800
HEIGHT = 600

def display() -> None:
    global DELTA_Z
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    
    glScalef(0.25, 0.25, 0.25)
    glTranslatef(0.0, -0.5, 0)
    glRotatef(45, 1, 0, 0)
    
    glPushMatrix()
    z = 0 - DELTA_Z
    print(z)
    # glTranslate(0, 0, z)
    models.draw_model(0, z)

    glPopMatrix()
    time.sleep(1)
            
    glutSwapBuffers()

def update(value: int) -> None:
    """Refreshes the window.

    Args:
        value: idk
    """
    glutPostRedisplay()
    glutTimerFunc(1000 // 60, update, 0)
    
def move_backwards():
    global DELTA_Z
    
    for i in range(40):
        DELTA_Z += 1
        glutPostRedisplay()
        time.sleep(1)

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
    threading.Thread(target=move_backwards).start()
    glutMainLoop()

if __name__ == "__main__":
    try:
        main()
    except:
        print("Program terminated")
        sys.exit()