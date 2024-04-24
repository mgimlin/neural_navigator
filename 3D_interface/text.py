from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

WIDTH = 800
HEIGHT = 600

vertices = (
    (1, -1, -1), (1, 1, -1),
    (-1, 1, -1), (-1, -1, -1),
    (1, -1, 1), (1, 1, 1),
    (-1, -1, 1), (-1, 1, 1)
)

faces = (
    (0, 1, 2, 3),
    (3, 2, 7, 6),
    (6, 7, 5, 4),
    (4, 5, 1, 0),
    (1, 5, 7, 2),
    (4, 0, 3, 6)
)

colors = (
    (1, 0, 0), (0, 1, 0), (0, 0, 1),
    (1, 1, 0), (1, 0, 1), (0, 1, 1)
)

def cube():
    glBegin(GL_QUADS)
    for i, face in enumerate(faces):
        glColor3fv(colors[i])
        for vertex in face:
            glVertex3fv(vertices[vertex])
    glEnd()

def draw_text(x, y, text):
    glWindowPos2f(x, y)
    for char in text:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))

def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluPerspective(45, (WIDTH / HEIGHT), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -10)

    positions = [(-2, 0, 0), (0, 2, 0), (2, 0, 0)]
    labels = ["Cube 1", "Cube 2", "Cube 3"]

    for i, pos in enumerate(positions):
        glPushMatrix()
        glTranslatef(*pos)
        glRotatef(glutGet(GLUT_ELAPSED_TIME) / 1000.0 * 45, 0, 1, 0)
        cube()
        glPopMatrix()

        # Projection of 3D coordinates above the cube to 2D screen coordinates
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        viewport = glGetIntegerv(GL_VIEWPORT)
        x2D, y2D, z2D = gluProject(pos[0], pos[1] + 2, pos[2], modelview, projection, viewport)

        # Draw the label at projected 2D position
        draw_text(x2D, viewport[3] - y2D, labels[i])  # Adjust for y-coordinate from bottom-left

    glutSwapBuffers()

def update(value):
    glutPostRedisplay()
    glutTimerFunc(int(1000 / 60), update, 0)

def main():
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(WIDTH, HEIGHT)
    glutCreateWindow("Multiple Cubes with Labels")
    glutDisplayFunc(display)
    glutTimerFunc(int(1000 / 60), update, 0)
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glEnable(GL_DEPTH_TEST)
    glutMainLoop()

if __name__ == "__main__":
    main()
