from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

import pywavefront

class OBJ:
    def __init__(self, object_id, name, scale, rotation, file_path):
        self.object_id = object_id
        self.name = name
        self.scale = scale
        self.rotation = rotation
        self.file_path = file_path
        self.model = None
        
    def preload_model(self):
        try: 
            self.model = pywavefront.Wavefront(self.file_path, collect_faces=True)
        except Exception as e:
            print(f'Error with {self.name}: {e}')
            
    def apply_scale(self):
        glScalef(*self.scale)
        
    def apply_rotation(self):
        glRotatef(*self.rotation)
        
    def get_scale_factor(self):
        return 1.0 / self.scale[2]
            
    def get_model(self):
        return self.model

objs = {
    0: OBJ(0, 'Person', (0.6608968783038234, 0.6608968783038234, 0.6608968783038234), (0, 0, 0, 0), 'objects/BaseMesh.obj'),
    1: OBJ(1, 'Bike', (0.015531929764613605, 0.015531929764613605, 0.015531929764613605), (90, -1, 0, 0), 'objects/bike.obj'),
    2: OBJ(2, 'Car', (0.49350982271093913, 0.49350982271093913, 0.49350982271093913), (0, 0, 0, 0), 'objects/car.obj'),
    3: OBJ(3, 'Motorcycle', (0.12066774731874456, 0.12066774731874456, 0.12066774731874456), (90, 0, -1, 0), 'objects/motorcycle.obj'),
    4: OBJ(4, 'Bus', (0.6889356709002767, 0.6889356709002767, 0.6889356709002767), (0, 0, 0, 0), 'objects/bus.obj'),
    5: OBJ(5, 'Truck', (0.5011002758977943, 0.5011002758977943, 0.5011002758977943), (0, 0, 0, 0), 'objects/truck.obj'),
    6: OBJ(6, 'Stop Sign', (0.21186699227515995, 0.21186699227515995, 0.21186699227515995), (90, 0, -1, 0), 'objects/StopSign.obj'),
    7: OBJ(7, 'Cat', (0.2839997629169979, 0.2839997629169979, 0.2839997629169979), (0, 0, 0, 0), 'objects/cat.obj'),
    8: OBJ(8, 'Dog', (0.023733508771430174, 0.023733508771430174, 0.023733508771430174), (0, 0, 0, 0), 'objects/dog.obj'),
    9: OBJ(9, 'Horse', (0.4578585187874002, 0.4578585187874002, 0.4578585187874002), (90, 0, -1, 0), 'objects/Horse.obj'),
    10: OBJ(10, 'Sheep', (0.8135220379052459, 0.8135220379052459, 0.8135220379052459), (90, 0, -1, 0), 'objects/sheep.obj'),
    11: OBJ(11, 'Cow', (0.0008176112303370176, 0.0008176112303370176, 0.0008176112303370176), (0, 0, 0, 0), 'objects/cow.obj'),
    12: OBJ(12, 'Frisbee', (0.021554883043204607, 0.021554883043204607, 0.021554883043204607), (0, 0, 0, 0), 'objects/frisbee.obj'),
    13: OBJ(13, 'Ball', (0.0025252525252525255, 0.0025252525252525255, 0.0025252525252525255), (0, 0, 0, 0), 'objects/ball.obj'),
    14: OBJ(14, 'Skateboard', (1.5706911543700397, 1.5706911543700397, 1.5706911543700397), (90, 0, -1, 0), 'objects/skateboard.obj'),
    15: OBJ(15, 'Green Traffic Light', (0.30378719342555904, 0.30378719342555904, 0.30378719342555904), (90, 0, -1, 0), 'objects/trafficLight.obj'),
    16: OBJ(16, 'Red Traffic Light', (0.30378719342555904, 0.30378719342555904, 0.30378719342555904), (90, 0, -1, 0), 'objects/trafficLight.obj'),
    17: OBJ(17, 'Yellow Traffic Light', (0.30378719342555904, 0.30378719342555904, 0.30378719342555904), (90, 0, -1, 0), 'objects/trafficLight.obj'),
    19: OBJ(19, 'Scooter', (0.5227721645987892, 0.5227721645987892, 0.5227721645987892), (0, 0, 0, 0), 'objects/scooter.obj'),
    21: OBJ(21, 'Cone', (1.0195531911476563, 1.0195531911476563, 1.0195531911476563), (0, 0, 0, 0), 'objects/cone.obj'),
    22: OBJ(22, 'Trash', (0.02237857385824516, 0.02237857385824516, 0.02237857385824516), (0, 0, 0, 0), 'objects/trash.obj'),
}

def preload_models() -> None:
    for obj in objs.values():
        obj.preload_model()

def draw_model(object_id: int, z: int) -> None:
    obj = objs.get(object_id)

    obj.apply_scale()
    glTranslatef(0, -0.5, z*obj.get_scale_factor())
    obj.apply_rotation()

    
    model = obj.get_model()
    
    glColor3f(.75, 0.75, 0.75)
    
    for mesh in model.mesh_list:
        glBegin(GL_TRIANGLES)
        for face in mesh.faces:
            for vertex_index in face:
                glVertex3f(*model.vertices[vertex_index])        
        glEnd()