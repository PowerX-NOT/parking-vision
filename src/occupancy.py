from slot_detection import SLOT_RECTS

def overlaps(boxA, boxB):
    # Returns True if two rectangles overlap
    ax1, ay1, aw, ah = boxA
    bx1, by1, bw, bh = boxB
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    # Compute overlap
    overlap_x = max(0, min(ax2, bx2) - max(ax1, bx1))
    overlap_y = max(0, min(ay2, by2) - max(ay1, by1))
    return (overlap_x > 0) and (overlap_y > 0)

def classify_occupancy(frame, car_boxes):
    statuses = []
    for slot in SLOT_RECTS:
        occupied = any(overlaps(slot, car) for car in car_boxes)
        statuses.append('occupied' if occupied else 'vacant')
    return statuses
