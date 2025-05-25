import numpy as np

def detect_vehicles(image, model, conf=0.4):
    """
    Run YOLO detection and return bounding boxes for vehicles only.
    Returns: list of rectangles [(x1, y1, x2, y2), ...]
    """
    results = model(image)
    vehicles = []
    vehicle_classes = {'car', 'truck', 'bus', 'motorcycle', 'van'}  # adjust as needed
    for result in results:
        names = result.names
        if getattr(result, "boxes", None) is not None:
            xyxy = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            for i, box in enumerate(xyxy):
                if confs[i] >= conf and names[classes[i]] in vehicle_classes:
                    x1, y1, x2, y2 = box.astype(int)
                    vehicles.append(((x1, y1), (x2, y2)))
    return vehicles

def rect_overlap(r1, r2, threshold=0.2):
    """
    Returns True if r1 and r2 overlap more than threshold IoU.
    r1, r2: ((x1, y1), (x2, y2))
    """
    xA = max(r1[0][0], r2[0][0])
    yA = max(r1[0][1], r2[0][1])
    xB = min(r1[1][0], r2[1][0])
    yB = min(r1[1][1], r2[1][1])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = abs((r1[1][0] - r1[0][0]) * (r1[1][1] - r1[0][1]))
    boxBArea = abs((r2[1][0] - r2[0][0]) * (r2[1][1] - r2[0][1]))

    if boxAArea == 0 or boxBArea == 0:
        return False

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou > threshold

def check_slots_occupancy(slots, vehicles):
    """
    For each slot, check if any vehicle overlaps it.
    Returns: list of True (occupied) or False (vacant)
    """
    statuses = []
    for slot in slots:
        occupied = any(rect_overlap(slot, veh) for veh in vehicles)
        statuses.append(occupied)
    return statuses
