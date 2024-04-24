
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import pywavefront

# Load the OBJ file (replace 'your_model.o
# bj' with the actual filename)
scene = pywavefront.Wavefront('objs/sheep.obj', collect_faces=True)
#bike needs to be rotated 90 degrees along the x axis 
#dog needs to be rotated
#motorbike neds to be rotated
#bus needs to be rotated
#stop sign needs to be rotated

# Compute the scene bounding box
scene_box = (scene.vertices[0], scene.vertices[0])
for vertex in scene.vertices:
    min_v = [min(scene_box[0][i], vertex[i]) for i in range(3)]
    max_v = [max(scene_box[1][i], vertex[i]) for i in range(3)]
    scene_box = (min_v, max_v)

# Compute translation and scale factors
scene_trans = [-(scene_box[1][i] + scene_box[0][i]) / 2 for i in range(3)]
scaled_size = 5
scene_size = [scene_box[1][i] - scene_box[0][i] for i in range(3)]
max_scene_size = max(scene_size)
scene_scale = [scaled_size / max_scene_size for i in range(3)]

def Model():
    glPushMatrix()
    glScalef(*scene_scale)
    glTranslatef(*scene_trans)

    glColor3f(0.5, 0.5, 0.5)  # Set the color to red

    for mesh in scene.mesh_list:
        glBegin(GL_TRIANGLES)
        for face in mesh.faces:
            for vertex_i in face:
                glVertex3f(*scene.vertices[vertex_i])
        glEnd()

    glPopMatrix()

def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    gluPerspective(45, (display[0] / display[1]), 1, 500.0)
    glTranslatef(0.0, 0.0, -10)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    glTranslatef(-0.5, 0, 0)
                if event.key == pygame.K_RIGHT:
                    glTranslatef(0.5, 0, 0)
                if event.key == pygame.K_UP:
                    glTranslatef(0, 1, 0)
                if event.key == pygame.K_DOWN:
                    glTranslatef(0, -1, 0)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        Model()
        pygame.display.flip()
        pygame.time.wait(10)

if __name__ == "__main__":
    main()
