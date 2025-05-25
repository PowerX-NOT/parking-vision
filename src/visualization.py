import cv2

def visualize_results(image, slots, statuses, vehicles=None):
    """
    Draws slots, vehicle boxes, and status labels.
    """
    output = image.copy()
    vacant = 0
    occupied = 0
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
    if vehicles:
        for veh in vehicles:
            cv2.rectangle(output, veh[0], veh[1], (255, 255, 0), 2)
    cv2.putText(output, f"Vacant: {vacant}  Occupied: {occupied}", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    return output
