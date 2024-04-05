from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

# Window dimensions
WIDTH = 800
HEIGHT = 600

# Define vertices of the cube
vertices= (
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, -1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, -1, 1),
    (-1, 1, 1),
)

# Define the edges connecting the vertices
edges = (
    (0, 1),
    (0, 3),
    (0, 4),
    (2, 1),
    (2, 3),
    (2, 7),
    (6, 3),
    (6, 4),
    (6, 7),
    (5, 1),
    (5, 4),
    (5, 7),
)

def cube() -> None:
    # Begin drawing lines
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()

def display() -> None:
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    # Adjust the view
    gluPerspective(45, (WIDTH/HEIGHT), 0.1, 50.0)
    glTranslatef(0.0,0.0, -5)

    cube()

    glutSwapBuffers()

def update(value) -> None:
    glutPostRedisplay()
    glutTimerFunc(int(1000/60), update, 0)

def main() -> None:
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(WIDTH, HEIGHT)
    glutCreateWindow("3D Cube in PyOpenGL")
    glutDisplayFunc(display)
    glutTimerFunc(int(1000/60), update, 0)
    # Setup perspective projection and depth testing for 3D
    glEnable(GL_DEPTH_TEST)
    glutMainLoop()

if __name__ == "__main__":
    main()
