import pywavefront

model_files = {
    'person': 'objects/BaseMesh.obj',
    'bike': 'objects/bike.obj',
    'car': 'objects/car.obj',
    'motorcycle': 'objects/motorcycle.obj',
    'truck': 'objects/truck.obj',
    'stopSign': 'objects/StopSign.obj',
    'cat': 'objects/cat.obj',
    'dog': 'objects/dog.obj',
    'horse': 'objects/Horse.obj',
    'sheep': 'objects/sheep.obj',
    'cow': 'objects/cow.obj',
    'frisbee': 'objects/frisbee.obj',
    'ball': 'objects/ball.obj',
    'skateboard': 'objects/skateboard.obj',
    'trafficLight': 'objects/trafficLight.obj',
    'scooter': 'objects/scooter.obj',
    'truck': 'objects/truck.obj',
    'cone': 'objects/cone.obj',
    'trash': 'objects/trash.obj'
}

def preload_models():
    global model_files
    models = {}
    for key, file_path in model_files.items():
        try:
            models[key] = pywavefront.Wavefront(file_path, collect_faces=True)
            print(f"{key} loaded")
        except Exception as e:
            print(f'Error with {key}: {e}')
            
    return models