# # import glfw
# # from OpenGL.GL import *
# # from OpenGL.GL.shaders import compileShader, compileProgram
# # import numpy as np
# # import glm

# # # Define constants
# # WINDOW_WIDTH = 800
# # WINDOW_HEIGHT = 600

# # # Define the Light class
# # class Light:
# #     def __init__(self, position, color):
# #         self.position = position
# #         self.color = color

# # # Define the Material class
# # class Material:
# #     def __init__(self, ka, kd, ks, s):
# #         self.ka = ka
# #         self.kd = kd
# #         self.ks = ks
# #         self.s = s

# # # Initialize GLFW
# # glfw.init()
# # glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
# # window = glfw.create_window(WINDOW_WIDTH, WINDOW_HEIGHT, "Bunny Rendering", None, None)
# # glfw.make_context_current(window)

# # def load_obj(filename):
# #     vertices = []
# #     faces = []
# #     with open(filename, 'r') as f:
# #         for line in f:
# #             if line.startswith('v '):
# #                 vertices.append(list(map(float, line.strip().split()[1:])))
# #             elif line.startswith('f '):
# #                 faces.append([int(vertex.split('/')[0]) - 1 for vertex in line.strip().split()[1:]])
# #     return vertices, faces

# # # Load shader source code from external files
# # def load_shader_source(shader_file):
# #     with open(shader_file, 'r') as f:
# #         return f.read()

# # vertex_shader_src = load_shader_source('vert2.glsl')
# # fragment_shader_src = load_shader_source('frag2.glsl')

# # # Compile shaders
# # vertex_shader = compileShader(vertex_shader_src, GL_VERTEX_SHADER)
# # fragment_shader = compileShader(fragment_shader_src, GL_FRAGMENT_SHADER)

# # # Create shader program and link shaders
# # shader_program = compileProgram(vertex_shader, fragment_shader)
# # glUseProgram(shader_program)

# # # Set up vertex data (bunny vertices, normals, etc.)
# # # You should replace this with your actual vertex data loading code
# # # For simplicity, we assume you have vertex and normal data loaded here
# # vertices = np.array([...], dtype=np.float32)  # Replace [...] with actual vertex data
# # normals = np.array([...], dtype=np.float32)  # Replace [...] with actual normal data

# # # Vertex Buffer Objects (VBOs) setup
# # vbo_vertices = glGenBuffers(1)
# # glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices)
# # glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

# # vbo_normals = glGenBuffers(1)
# # glBindBuffer(GL_ARRAY_BUFFER, vbo_normals)
# # glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)

# # # Vertex Array Object (VAO) setup
# # vao = glGenVertexArrays(1)
# # glBindVertexArray(vao)

# # # Bind vertices
# # glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices)
# # glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
# # glEnableVertexAttribArray(0)

# # # Bind normals
# # glBindBuffer(GL_ARRAY_BUFFER, vbo_normals)
# # glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
# # glEnableVertexAttribArray(1)

# # # Set up matrices and uniform locations
# # model_loc = glGetUniformLocation(shader_program, "model")
# # view_loc = glGetUniformLocation(shader_program, "view")
# # projection_loc = glGetUniformLocation(shader_program, "projection")
# # eye_loc = glGetUniformLocation(shader_program, "eye")
# # spotlight_loc = glGetUniformLocation(shader_program, "spotlight")
# # ka_loc = glGetUniformLocation(shader_program, "ka")
# # kd_loc = glGetUniformLocation(shader_program, "kd")
# # ks_loc = glGetUniformLocation(shader_program, "ks")
# # s_loc = glGetUniformLocation(shader_program, "s")

# # # Define light properties and create light objects
# # lights = [
# #     Light(position=glm.vec3(0.0, 0.0, 3.0), color=glm.vec3(0.5, 0.5, 0.5)),
# #     Light(position=glm.vec3(0.0, 3.0, 0.0), color=glm.vec3(0.2, 0.2, 0.2))
# # ]

# # # Define material properties and create material object
# # material = Material(
# #     ka=glm.vec3(0.2, 0.2, 0.2),
# #     kd=glm.vec3(0.8, 0.7, 0.7),
# #     ks=glm.vec3(1.0, 1.0, 1.0),
# #     s=10.0
# # )

# # # Enable depth testing
# # glEnable(GL_DEPTH_TEST)

# # # Set projection matrix
# # projection = glm.perspective(glm.radians(45.0), WINDOW_WIDTH / WINDOW_HEIGHT, 0.1, 100.0)

# # # Main render loop
# # while not glfw.window_should_close(window):
# #     glfw.poll_events()

# #     # Clear buffers
# #     glClearColor(0.2, 0.3, 0.3, 1.0)
# #     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

# #     # Set view matrix and eye position
# #     view = glm.lookAt(glm.vec3(0.0, 0.0, 4.0), glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 1.0, 0.0))
# #     glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(view))
# #     glUniform3fv(eye_loc, 1, glm.value_ptr(glm.vec3(0.0, 0.0, 4.0)))

# #     # Activate shader program
# #     glUseProgram(shader_program)

# #     # Set uniform values
# #     glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(glm.mat4(1.0)))
# #     glUniformMatrix4fv(projection_loc, 1, GL_FALSE, glm.value_ptr(projection))
# #     glUniform1i(spotlight_loc, 0)  # Change to 1 to enable spotlight effect

# #     glUniform3fv(ka_loc, 1, glm.value_ptr(material.ka))
# #     glUniform3fv(kd_loc, 1, glm.value_ptr(material.kd))
# #     glUniform3fv(ks_loc, 1, glm.value_ptr(material.ks))
# #     glUniform1f(s_loc, material.s)

# #     # Bind VAO and draw
# #     glBindVertexArray(vao)
# #     glDrawArrays(GL_TRIANGLES, 0, len(vertices))

# #     # Swap buffers and poll events
# #     glfw.swap_buffers(window)

# # glfw.terminate()


# import glfw
# from OpenGL.GL import *
# from OpenGL.GL.shaders import compileShader, compileProgram
# import numpy as np
# import glm

# # Define constants
# WINDOW_WIDTH = 800
# WINDOW_HEIGHT = 600

# # Define the Light class
# class Light:
#     def __init__(self, position, color):
#         self.position = position
#         self.color = color

# # Define the Material class
# class Material:
#     def __init__(self, ka, kd, ks, s):
#         self.ka = ka
#         self.kd = kd
#         self.ks = ks
#         self.s = s

# # Initialize GLFW
# glfw.init()
# glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
# window = glfw.create_window(WINDOW_WIDTH, WINDOW_HEIGHT, "Bunny Rendering", None, None)
# glfw.make_context_current(window)
# glfw.swap_interval(1)  # Enable VSync

# # Load shader source code from external files
# def load_shader_source(shader_file):
#     with open(shader_file, 'r') as f:
#         return f.read()

# vertex_shader_src = load_shader_source('vert2.glsl')
# fragment_shader_src = load_shader_source('frag2.glsl')

# # Compile shaders
# vertex_shader = compileShader(vertex_shader_src, GL_VERTEX_SHADER)
# fragment_shader = compileShader(fragment_shader_src, GL_FRAGMENT_SHADER)

# # Create shader program and link shaders
# shader_program = compileProgram(vertex_shader, fragment_shader)
# glUseProgram(shader_program)


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
#                 face = [list(map(int, vertex.split('//'))) for vertex in line.strip().split()[1:]]
#                 faces.append(face)
#     return vertices, normals, faces

# # Load OBJ file
# vertices, normals, faces = load_obj('bunny.obj')


# # Convert faces to flat list of indices
# indices = [vertex_index for face in faces for vertex_indices in face for vertex_index in vertex_indices]

# vertices_array = np.array(vertices, dtype=np.float32)
# normals_array = np.array(normals, dtype=np.float32)

# # Vertex Buffer Objects (VBOs) setup
# vbo_vertices = glGenBuffers(1)
# glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices)
# glBufferData(GL_ARRAY_BUFFER, vertices_array.nbytes, vertices_array, GL_STATIC_DRAW)

# vbo_normals = glGenBuffers(1)
# glBindBuffer(GL_ARRAY_BUFFER, vbo_normals)
# glBufferData(GL_ARRAY_BUFFER, normals_array.nbytes, normals_array, GL_STATIC_DRAW)

# # Vertex Array Object (VAO) setup
# vao = glGenVertexArrays(1)
# glBindVertexArray(vao)

# # Bind vertices
# glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices)
# glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
# glEnableVertexAttribArray(0)

# # Bind normals
# glBindBuffer(GL_ARRAY_BUFFER, vbo_normals)
# glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
# glEnableVertexAttribArray(1)

# # Set up matrices and uniform locations
# model_loc = glGetUniformLocation(shader_program, "model")
# view_loc = glGetUniformLocation(shader_program, "view")
# projection_loc = glGetUniformLocation(shader_program, "projection")
# eye_loc = glGetUniformLocation(shader_program, "eye")
# spotlight_loc = glGetUniformLocation(shader_program, "spotlight")
# ka_loc = glGetUniformLocation(shader_program, "ka")
# kd_loc = glGetUniformLocation(shader_program, "kd")
# ks_loc = glGetUniformLocation(shader_program, "ks")
# s_loc = glGetUniformLocation(shader_program, "s")

# # Define light properties and create light objects
# lights = [
#     Light(position=glm.vec3(0.0, 0.0, 3.0), color=glm.vec3(0.5, 0.5, 0.5)),
#     Light(position=glm.vec3(0.0, 3.0, 0.0), color=glm.vec3(0.2, 0.2, 0.2))
# ]

# # Define material properties and create material object
# material = Material(
#     ka=glm.vec3(0.2, 0.2, 0.2),
#     kd=glm.vec3(0.8, 0.7, 0.7),
#     ks=glm.vec3(1.0, 1.0, 1.0),
#     s=10.0
# )

# # Enable depth testing
# glEnable(GL_DEPTH_TEST)

# # Set projection matrix
# projection = glm.perspective(glm.radians(45.0), WINDOW_WIDTH / WINDOW_HEIGHT, 0.1, 100.0)

# # Main render loop
# while not glfw.window_should_close(window):
#     glfw.poll_events()

#     # Clear buffers
#     glClearColor(0.2, 0.3, 0.3, 1.0)
#     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

#     # Set view matrix and eye position
#     view = glm.lookAt(glm.vec3(0.0, 0.0, 4.0), glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 1.0, 0.0))
#     glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(view))
#     glUniform3fv(eye_loc, 1, glm.value_ptr(glm.vec3(0.0, 0.0, 4.0)))

#     # Activate shader program
#     glUseProgram(shader_program)

#     # Set uniform values
#     glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(glm.mat4(1.0)))
#     glUniformMatrix4fv(projection_loc, 1, GL_FALSE, glm.value_ptr(projection))
#     glUniform1i(spotlight_loc, 0)  # Change to 1 to enable spotlight effect

#     glUniform3fv(ka_loc, 1, glm.value_ptr(material.ka))
#     glUniform3fv(kd_loc, 1, glm.value_ptr(material.kd))
#     glUniform3fv(ks_loc, 1, glm.value_ptr(material.ks))
#     glUniform1f(s_loc, material.s)

#     # Bind VAO and draw
#     glBindVertexArray(vao)
#     glDrawArrays(GL_TRIANGLES, 0, len(vertices))

#     # Swap buffers and poll events
#     glfw.swap_buffers(window)

# glfw.terminate()
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram
import numpy as np
import glm

# Define constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# Define vertex shader code
vertex_shader_code = """
#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

out vec3 FragPos;
out vec3 Normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
}
"""

# Define fragment shader code
fragment_shader_code = """
#version 330 core

in vec3 FragPos;
in vec3 Normal;

out vec4 FragColor;

uniform vec3 lightPos;
uniform vec3 lightColor;
uniform vec3 objectColor;

void main()
{
    // Ambient
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;

    // Diffuse
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    // Specular
    float specularStrength = 0.5;
    vec3 viewDir = normalize(-FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;

    vec3 result = (ambient + diffuse + specular) * objectColor;
    FragColor = vec4(result, 1.0);
}
"""

# Initialize GLFW
glfw.init()
glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
window = glfw.create_window(WINDOW_WIDTH, WINDOW_HEIGHT, "OBJ Shading", None, None)
glfw.make_context_current(window)

# Compile shaders
vertex_shader = compileShader(vertex_shader_code, GL_VERTEX_SHADER)
fragment_shader = compileShader(fragment_shader_code, GL_FRAGMENT_SHADER)

# Create shader program and link shaders
shader_program = compileProgram(vertex_shader, fragment_shader)
glUseProgram(shader_program)

# Set up vertex data (replace this with your actual OBJ loading code)
# Example vertices and normals (cube)
vertices = np.array([
    -0.5, -0.5, -0.5,
     0.5, -0.5, -0.5,
     0.5,  0.5, -0.5,
    -0.5,  0.5, -0.5,
    -0.5, -0.5,  0.5,
     0.5, -0.5,  0.5,
     0.5,  0.5,  0.5,
    -0.5,  0.5,  0.5
], dtype=np.float32)

normals = np.array([
    0, 0, -1,
    0, 0, -1,
    0, 0, -1,
    0, 0, -1,
    0, 0, 1,
    0, 0, 1,
    0, 0, 1,
    0, 0, 1
], dtype=np.float32)

def load_obj(filename):
    vertices = []
    normals = []
    faces = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertices.append(list(map(float, line.strip().split()[1:])))
            elif line.startswith('vn '):
                normals.append(list(map(float, line.strip().split()[1:])))
            elif line.startswith('f '):
                face = [list(map(int, vertex.split('//'))) for vertex in line.strip().split()[1:]]
                faces.append(face)
    return vertices, normals, faces

# Load OBJ file
vertices, normals, faces = load_obj('bunny.obj')


# Convert faces to flat list of indices
indices = [vertex_index for face in faces for vertex_indices in face for vertex_index in vertex_indices]

vertices = np.array(vertices, dtype=np.float32)
normals = np.array(normals, dtype=np.float32)

# Vertex Buffer Objects (VBOs) setup
vbo_vertices = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices)
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

vbo_normals = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vbo_normals)
glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)

# Vertex Array Object (VAO) setup
vao = glGenVertexArrays(1)
glBindVertexArray(vao)

# Bind vertices
glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
glEnableVertexAttribArray(0)

# Bind normals
glBindBuffer(GL_ARRAY_BUFFER, vbo_normals)
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
glEnableVertexAttribArray(1)

# Set uniform locations
model_loc = glGetUniformLocation(shader_program, "model")
view_loc = glGetUniformLocation(shader_program, "view")
projection_loc = glGetUniformLocation(shader_program, "projection")
light_pos_loc = glGetUniformLocation(shader_program, "lightPos")
light_color_loc = glGetUniformLocation(shader_program, "lightColor")
object_color_loc = glGetUniformLocation(shader_program, "objectColor")

# Set shader uniforms
glUniform3f(light_pos_loc, 1.0, 2.0, 3.0)  # Example light position
glUniform3f(light_color_loc, 1.0, 1.0, 1.0)  # White light
glUniform3f(object_color_loc, 1.0, 0.5, 0.31)  # Orange object color

# Enable depth testing
glEnable(GL_DEPTH_TEST)

# Main render loop
while not glfw.window_should_close(window):
    glfw.poll_events()

    # Clear buffers
    glClearColor(0.2, 0.3, 0.3, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Set view and projection matrices (you can modify these as needed)
    view = glm.lookAt(glm.vec3(0, 0, 3), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
    projection = glm.perspective(glm.radians(45.0), WINDOW_WIDTH / WINDOW_HEIGHT, 0.1, 100.0)

    glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(view))
    glUniformMatrix4fv(projection_loc, 1, GL_FALSE, glm.value_ptr(projection))

    # Render
    model = glm.mat4(1.0)
    model = glm.rotate(model, glm.radians(glfw.get_time() * 50), glm.vec3(0.5, 1.0, 0.0))  # Rotate model
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(model))

    glBindVertexArray(vao)
    glDrawArrays(GL_TRIANGLES, 0, len(vertices) // 3)

    # Swap buffers and poll events
    glfw.swap_buffers(window)

glfw.terminate()

