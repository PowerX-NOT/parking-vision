import cv2
import numpy as np

def visualize_results(image, slots, statuses, obb_polygons=None):
    """
    Draws slots and optionally detected object polygons.
    If a vehicle polygon overlaps a slot, only the red slot is shown (not the vehicle polygon).
    Otherwise, vehicle polygons (not in slots) are drawn in blue.
    """
    output = image.copy()
    vacant = 0
    occupied = 0

    # Prepare slot polygons for overlap check
    slot_polygons = []
    for pt1, pt2 in slots:
        # Rectangle to polygon: top-left, bottom-left, bottom-right, top-right
        x1, y1 = pt1
        x2, y2 = pt2
        slot_poly = np.array([
            [x1, y1],
            [x1, y2],
            [x2, y2],
            [x2, y1]
        ], dtype=np.int32)
        slot_polygons.append(slot_poly)

    # Draw slots
    for i, rect in enumerate(slots):
        color = (0, 255, 0) if not statuses[i] else (0, 0, 255)
        cv2.rectangle(output, rect[0], rect[1], color, 2)
        label = "Vacant" if not statuses[i] else "Occupied"
        cv2.putText(output, label, (rect[0][0], rect[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if not statuses[i]:
            vacant += 1
        else:
            occupied += 1

    # Draw vehicle polygons ONLY if they are not inside any slot
    if obb_polygons:
        for veh in obb_polygons:
            veh_pts = veh.astype(np.int32).reshape((-1, 1, 2))
            # Check overlap with any slot polygon
            in_slot = False
            for slot_poly in slot_polygons:
                slot_pts = slot_poly.reshape((-1, 1, 2))
                # Use cv2.intersectConvexConvex to check intersection area
                inter_area, _ = cv2.intersectConvexConvex(veh.astype(np.float32), slot_poly.astype(np.float32))
                veh_area = cv2.contourArea(veh.astype(np.float32))
                if veh_area > 0 and (inter_area / veh_area) > 0.2:
                    in_slot = True
                    break
            if not in_slot:
                cv2.polylines(output, [veh_pts], isClosed=True, color=(255, 0, 0), thickness=2)  # Blue for "not in slot"
            # If in_slot, do NOT draw the vehicle polygon, just let the slot box be red

    cv2.putText(output, f"Vacant: {vacant}  Occupied: {occupied}", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    return output
