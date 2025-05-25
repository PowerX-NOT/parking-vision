import cv2
import numpy as np

def visualize_results(image, slots, statuses, obb_polygons=None):
    """
    Draws colored rectangles for slots (red=occupied, green=vacant) and vehicle polygons.
    If a vehicle polygon overlaps a slot, only the red slot is shown (not the vehicle polygon).
    Returns the main image and a summary counts image for display in a separate window.
    """
    output = image.copy()
    vacant = 0
    occupied = 0

    # Prepare slot polygons for overlap check
    slot_polygons = []
    for pt1, pt2 in slots:
        x1, y1 = pt1
        x2, y2 = pt2
        slot_poly = np.array([
            [x1, y1],
            [x1, y2],
            [x2, y2],
            [x2, y1]
        ], dtype=np.int32)
        slot_polygons.append(slot_poly)

    # Draw slots (no per-slot labels)
    for i, rect in enumerate(slots):
        color = (0, 255, 0) if not statuses[i] else (0, 0, 255)
        cv2.rectangle(output, rect[0], rect[1], color, 2)
        if not statuses[i]:
            vacant += 1
        else:
            occupied += 1

    # Draw vehicle polygons ONLY if they are not inside any slot
    if obb_polygons:
        for veh in obb_polygons:
            veh_pts = veh.astype(np.int32).reshape((-1, 1, 2))
            in_slot = False
            for slot_poly in slot_polygons:
                inter_area, _ = cv2.intersectConvexConvex(veh.astype(np.float32), slot_poly.astype(np.float32))
                veh_area = cv2.contourArea(veh.astype(np.float32))
                if veh_area > 0 and (inter_area / veh_area) > 0.2:
                    in_slot = True
                    break
            if not in_slot:
                cv2.polylines(output, [veh_pts], isClosed=True, color=(255, 0, 0), thickness=2)

    total = len(slots)
    # Make a simple white rectangle image for the dialog
    dialog_h = 120
    dialog_w = 340
    dialog = np.ones((dialog_h, dialog_w, 3), dtype=np.uint8) * 255
    cv2.putText(dialog, f"Total: {total}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(dialog, f"Vacant: {vacant}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,128,0), 2)
    cv2.putText(dialog, f"Occupied: {occupied}", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    return output, dialog
