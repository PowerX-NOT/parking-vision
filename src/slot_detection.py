import numpy as np

def detect_objects_polygons(image, model, conf=0.4):
    """
    Detects objects in the image using the YOLO OBB model.
    Returns a list of polygons (numpy arrays of shape (4, 2)), one per detected object.
    """
    results = model(image)
    polygons = []
    for result in results:
        if getattr(result, "obb", None) is not None:
            obb_polys = result.obb.xyxyxyxy.cpu().numpy()
            obb_confs = result.obb.conf.cpu().numpy()
            for i, poly in enumerate(obb_polys):
                if obb_confs[i] >= conf:
                    # poly is shape (8,), reshape to (4,2)
                    polygons.append(poly.reshape((4, 2)))
    return polygons

def polygon_overlap(poly1, poly2, threshold=0.15):
    """
    Returns True if poly1 and poly2 overlap more than threshold IoU.
    Uses cv2.intersectConvexConvex for polygon intersection area.
    """
    import cv2
    poly1 = poly1.astype(np.float32)
    poly2 = poly2.astype(np.float32)
    # Calculate intersection area
    area, _ = cv2.intersectConvexConvex(poly1, poly2)
    area1 = cv2.contourArea(poly1)
    area2 = cv2.contourArea(poly2)
    union = area1 + area2 - area
    if union == 0:
        return False
    iou = area / union
    return iou > threshold

def check_slots_occupancy(slots, obb_polygons):
    """
    For each slot (as polygon), check if any detected object polygon overlaps it.
    Returns: list of True (occupied) or False (vacant)
    """
    statuses = []
    for slot in slots:
        slot_poly = np.array([slot[0], (slot[0][0], slot[1][1]), slot[1], (slot[1][0], slot[0][1])], dtype=np.float32)
        occupied = any(polygon_overlap(slot_poly, obb) for obb in obb_polygons)
        statuses.append(occupied)
    return statuses
