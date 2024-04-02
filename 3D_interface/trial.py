import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import sys

def load_obj(filename):
    vertices = []
    faces = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertices.append(list(map(float, line.strip().split()[1:])))
            elif line.startswith('f '):
                faces.append([int(vertex.split('/')[0]) - 1 for vertex in line.strip().split()[1:]])
    return vertices, faces

def draw_obj(vertices, faces):
    glBegin(GL_TRIANGLES)
    for face in faces:
        for vertex_index in face:
            glVertex3fv(vertices[vertex_index])
    glEnd()

def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)

    obj_vertices, obj_faces = load_obj('bunny.obj')

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glColor3f(1.0, 1.0, 1.0)  # Set object color to white
        draw_obj(obj_vertices, obj_faces)
        
        pygame.display.flip()
        pygame.time.wait(10)

if __name__ == "__main__":
    main()


# import pygame
# from pygame.locals import *
# from OpenGL.GL import *
# from OpenGL.GLUT import *
# from OpenGL.GLU import *
# import sys
# import glm

# def load_obj(filename):
#     vertices = []
#     normals = []
#     faces = []
#     with open(filename, 'r') as f:
#         for line in f:
#             if line.startswith('v '):
#                 vertices.append(list(map(float, line.strip().split()[1:])))
#             elif line.startswith('vn '):
#                 normals.append(list(map(float, line.strip().split()[1:])))
#             elif line.startswith('f '):
#                 faces.append([tuple(map(int, vertex.split('//'))) for vertex in line.strip().split()[1:]])
#     return vertices, normals, faces

# def compile_shader(shader_type, source):
#     shader = glCreateShader(shader_type)
#     glShaderSource(shader, source)
#     glCompileShader(shader)
#     if not glGetShaderiv(shader, GL_COMPILE_STATUS):
#         error = glGetShaderInfoLog(shader).decode()
#         print(f"Shader compilation failed: {error}")
#         glDeleteShader(shader)
#         return None
#     return shader

# def create_shader_program(vertex_shader_source, fragment_shader_source):
#     vertex_shader = compile_shader(GL_VERTEX_SHADER, vertex_shader_source)
#     fragment_shader = compile_shader(GL_FRAGMENT_SHADER, fragment_shader_source)
#     if not (vertex_shader and fragment_shader):
#         return None
#     shader_program = glCreateProgram()
#     glAttachShader(shader_program, vertex_shader)
#     glAttachShader(shader_program, fragment_shader)
#     glLinkProgram(shader_program)
#     if not glGetProgramiv(shader_program, GL_LINK_STATUS):
#         error = glGetProgramInfoLog(shader_program).decode()
#         print(f"Shader program linking failed: {error}")
#         glDeleteProgram(shader_program)
#         return None
#     return shader_program

# def draw_obj(vertices, normals, faces):
#     glBegin(GL_TRIANGLES)
#     for face in faces:
#         for vertex_index, normal_index in face:
#             glNormal3fv(normals[normal_index - 1])
#             glVertex3fv(vertices[vertex_index - 1])
#     glEnd()

# def main():
#     pygame.init()
#     display = (800, 600)
#     pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

#     gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
#     glTranslatef(0.0, 0.0, -5)

#     obj_vertices, obj_normals, obj_faces = load_obj('bunny.obj')

#     shader_program = create_shader_program(
#         open('vert2.glsl').read(),
#         open('frag2.glsl').read()
#     )
#     glUseProgram(shader_program)

#     while True:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 pygame.quit()
#                 sys.exit()

#         glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

#         # Set view and projection matrices (you may need to modify these based on your scene)
#         view = glm.lookAt(glm.vec3(0, 0, 5), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
#         projection = glm.perspective(glm.radians(45), display[0] / display[1], 0.1, 100.0)
#         glUniformMatrix4fv(glGetUniformLocation(shader_program, "view"), 1, GL_FALSE, glm.value_ptr(view))
#         glUniformMatrix4fv(glGetUniformLocation(shader_program, "projection"), 1, GL_FALSE, glm.value_ptr(projection))

#         # Set light and material properties here as uniforms

#         # Draw the object
#         glColor3f(1.0, 1.0, 1.0)
#         draw_obj(obj_vertices, obj_normals, obj_faces)

#         pygame.display.flip()
#         pygame.time.wait(10)

# if __name__ == "__main__":
#     main()

