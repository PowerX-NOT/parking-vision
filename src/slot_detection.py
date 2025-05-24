from roboflow import Roboflow

def load_model(api_key, workspace, project_name, version):
    rf = Roboflow(api_key=api_key)
    project = rf.workspace().project(project_name)
    return project.version(version).model

def detect_slots(model, image_path, confidence, overlap):
    result = model.predict(image_path, confidence=confidence, overlap=overlap).json()
    return result["predictions"]