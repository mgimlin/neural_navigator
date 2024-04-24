
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from ultralytics import YOLO
import cv2
import torch
import numpy as np
from monodepth2 import monodepth2
import pywavefront

person = pywavefront.Wavefront('objects/BaseMesh.obj', collect_faces=True)
car = pywavefront.Wavefront('objects/car.obj', collect_faces=True)
cow = pywavefront.Wavefront('objects/cow.obj', collect_faces=True)
bus = pywavefront.Wavefront('objects/bus.obj', collect_faces=True)
dog = pywavefront.Wavefront('objects/dog.obj', collect_faces=True)
motorcycle = pywavefront.Wavefront('objects/motorcycle.obj', collect_faces=True)
stpsgn = pywavefront.Wavefront('objects/StopSign.obj', collect_faces=True)
bike = pywavefront.Wavefront('objects/bike.obj', collect_faces=True)
truck = pywavefront.Wavefront('objects/truck.obj', collect_faces=True)
cone = pywavefront.Wavefront('objects/cone.obj', collect_faces=True)




classDict = {
    0: person,        
    1: bike,       
    2: car,
    3: motorcycle,   
    4: bus,
    5: truck,
    6: stpsgn,    
    7: "cat",
    8: dog,
    9: "horse",
    10: "sheep",
    11: cow,
    12: "frisbee",
    13: "spball",
    14: "skateboard",
    15: "GTL",
    16: "RTL",
    17: "YTL",
    18: "construction",
    19: "scooter",
    20: "truck",
    21: cone,
    22: "trash",
    23: "car"
}
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = YOLO('best.pt').to(device)
cam = cv2.VideoCapture(0) 
if not cam.isOpened():
    exit()
    
# model_type = "MiDaS_small"
# midas = torch.hub.load("intel-isl/MiDaS", model_type)
# midas.to(device)
# midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
# transform = midas_transforms.small_transform
monodepth_model = monodepth2()




# # Compute the scene bounding box
# scene_box = (scene.vertices[0], scene.vertices[0])
# for vertex in scene.vertices:
#     min_v = [min(scene_box[0][i], vertex[i]) for i in range(3)]
#     max_v = [max(scene_box[1][i], vertex[i]) for i in range(3)]
#     scene_box = (min_v, max_v)

# # Compute translation and scale factors
# scene_trans = [-(scene_box[1][i] + scene_box[0][i]) / 2 for i in range(3)]
# scaled_size = 5
# scene_size = [scene_box[1][i] - scene_box[0][i] for i in range(3)]
# max_scene_size = max(scene_size)
# scene_scale = [scaled_size / max_scene_size for i in range(3)]


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

angle = 0  # Initial angle

def cube():
    glBegin(GL_QUADS)
    for i, face in enumerate(faces):
        glColor3fv(colors[i])  # Set color for each face
        for vertex in face:
            glVertex3fv(vertices[vertex])
    glEnd()

def Model(obj):
    # glPushMatrix()
    # glScalef(*scene_scale)
    # glTranslatef(*scene_trans)

    glColor3f(0.75, 0.75, 0.75)  # Set the color to red

    for mesh in obj.mesh_list:
        glBegin(GL_TRIANGLES)
        for face in mesh.faces:
            for vertex_i in face:
                glVertex3f(*obj.vertices[vertex_i])
        glEnd()


def display():
    global angle
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluPerspective(90, (WIDTH/HEIGHT), 0.1, 50.0)
    glTranslatef(0.0, -0.5, -5)
    glRotatef(45, 1, 0, 0)

    ret, frame = cam.read()
    if not ret:
        return

    # Object detection.
    results = model(frame)
    
     # Depth estimation
    depth_map = monodepth_model.eval(frame)
    depth_map_norm = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map)) * 255
    print(depth_map)
    depth_map_16 = (depth_map_norm * 65535).astype(np.uint16)
    # cv2.imwrite('out.png', depth_map_16)
    
    # Depth estimation.
    # img = frame
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # input_batch = transform(img).to(device)
    # with torch.no_grad():
    #     prediction = midas(input_batch)
    #     prediction = torch.nn.functional.interpolate(
    #         prediction.unsqueeze(1),
    #         size=img.shape[:2],
    #         mode="bicubic",
    #         align_corners=False,
    #     ).squeeze()
    # depth_map = prediction.cpu().numpy()

    for result in results:
        print(result)
        for i,box in enumerate(result.boxes.xyxyn):
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            cls = (result.boxes.cls)[i]
            print(int(cls.item()))
            
            
            # print(center_x, len(depth_map[0]), int(center_x * len(depth_map[0])))
            # depth = depth_map[int(center_x * len(depth_map))][int(center_y * len(depth_map[0]))]
            depth_vals = depth_map[int(center_y), int(center_x)]        #get rgb vals
            depth = 0.299*depth_vals[0] + 0.587*depth_vals[1] + 0.114*depth_vals[2]     #convert to singular grayscale values
            print(depth)
            glPushMatrix()
            glTranslatef(
                20 * center_x - 10,
                -0.5,
                # -5,
                0 - 10 * (1 - abs(box[1] - box[3]))
                # 5 - depth *0.5
            )
            Model(classDict[int(cls.item())])
            glPopMatrix()

    # Rotate the cube
    # glPushMatrix()
    # glRotatef(angle, 0, 1, 0)  # Rotate around the y-axis
    # cube()
    # glPopMatrix()

    # glPushMatrix()
    # glTranslatef(5, -0.5, -5)
    # cube()
    # glPopMatrix()
    glutSwapBuffers()

    # angle += 1  # Increment the angle for the next frame

def update(value):
    glutPostRedisplay()
    glutTimerFunc(int(1000/60), update, 0)

def main():
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(WIDTH, HEIGHT)
    glutCreateWindow("Rotating Cube")
    glutDisplayFunc(display)
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glutTimerFunc(int(1000/60), update, 0)
    glEnable(GL_DEPTH_TEST)
    glutMainLoop()

if __name__ == "__main__":
    main()