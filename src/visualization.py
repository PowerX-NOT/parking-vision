import cv2

def draw_slots(image, slot_status):
    img = image.copy()
    for slot in slot_status:
        x1, y1, x2, y2 = slot['bbox']
        color = (0, 0, 255) if slot['occupied'] else (0, 255, 0)
        label = "Occupied" if slot['occupied'] else "Vacant"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img

def display_count(image, slot_status):
    total = len(slot_status)
    occupied = sum(1 for s in slot_status if s['occupied'])
    vacant = total - occupied
    msg = f"Occupied: {occupied} | Vacant: {vacant}"
    cv2.putText(image, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    return image