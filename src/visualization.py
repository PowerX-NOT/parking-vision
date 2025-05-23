import cv2
from slot_detection import SLOT_RECTS

def draw_results(frame, car_boxes, statuses):
    # Draw slot rectangles
    for (slot, status) in zip(SLOT_RECTS, statuses):
        x, y, w, h = slot
        color = (0, 0, 255) if status == 'occupied' else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        cv2.putText(frame, status, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    # Draw detected cars (blue)
    for (x, y, w, h) in car_boxes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Count
    vacant_count = statuses.count('vacant')
    occupied_count = statuses.count('occupied')
    label = f"Occupied: {occupied_count} | Vacant: {vacant_count}"
    cv2.putText(frame, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    return frame

def display_frame(frame, is_image=False, delay=1):
    cv2.imshow("Parking Vision", frame)
    if is_image:
        key = cv2.waitKey(0)
    else:
        key = cv2.waitKey(delay)
    if key == 27:
        return False
    return True
