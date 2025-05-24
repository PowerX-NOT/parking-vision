def is_overlap(boxA, boxB):
    # boxA and boxB: (x1, y1, x2, y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return False
    # Optionally, require a minimum overlap ratio
    slotArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    overlap_ratio = interArea / float(slotArea)
    return overlap_ratio > 0.15  # threshold (adjust as needed)

def classify_occupancy(slots, vehicles):
    slot_status = []
    for slot in slots:
        occupied = any(is_overlap(slot, veh['bbox']) for veh in vehicles)
        slot_status.append({'bbox': slot, 'occupied': occupied})
    return slot_status
