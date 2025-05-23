import cv2
from config import COLOR_VACANT, COLOR_OCCUPIED

def draw_results(frame, slots, statuses):
    """
    Draw colored rectangles and counts on frame.
    """
    for (box, status) in zip(slots, statuses):
        x, y, w, h = box
        color = COLOR_OCCUPIED if status == 'occupied' else COLOR_VACANT
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    # Display counts
    vacant_count = statuses.count('vacant')
    occupied_count = statuses.count('occupied')
    label = f"Occupied: {occupied_count} | Vacant: {vacant_count}"
    cv2.putText(frame, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    return frame

def display_frame(frame, delay=1):
    """
    Show frame in a window.
    """
    cv2.imshow("Parking Vision", frame)
    key = cv2.waitKey(delay)
    if key == 27: # ESC to quit
        return False
    return True