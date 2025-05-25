import numpy as np

def detect_slots(image, model):
    """
    Detects parking slots in the image using the YOLO OBB model.
    Returns a list of polygons, one per slot.
    """
    results = model(image)
    slots = []
    for result in results:
        if getattr(result, "obb", None) is not None:
            polygons = result.obb.xyxyxyxy.cpu().numpy()
            slots.extend(polygons)
    return slots