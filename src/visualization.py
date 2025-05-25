import cv2
import numpy as np

def visualize_results(image, slots, statuses):
    """
    Draws colored polygons and counts of occupied/vacant slots.
    """
    output = image.copy()
    vacant = 0
    occupied = 0
    for i, polygon in enumerate(slots):
        pts = polygon.reshape((-1, 1, 2)).astype(int)
        color = (0, 255, 0) if not statuses[i] else (0, 0, 255)  # Green for vacant, Red for occupied
        cv2.polylines(output, [pts], isClosed=True, color=color, thickness=2)
        label = "Vacant" if not statuses[i] else "Occupied"
        cv2.putText(output, label, (int(polygon[0][0]), int(polygon[0][1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if not statuses[i]:
            vacant += 1
        else:
            occupied += 1
    # Draw count
    cv2.putText(output, f"Vacant: {vacant}  Occupied: {occupied}", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    return output