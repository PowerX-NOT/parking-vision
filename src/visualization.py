import cv2

def draw_boxes(image, predictions, occupancy, color_vacant, color_occupied):
    """
    Draw colored rectangles for each slot and display occupancy count.
    """
    vacant, occupied = 0, 0
    for pred in predictions:
        x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
        cls = pred['class']
        status = occupancy[cls]
        color = color_occupied if status == "occupied" else color_vacant
        if status == "occupied":
            occupied += 1
        else:
            vacant += 1
        cv2.rectangle(image, (x-w//2, y-h//2), (x+w//2, y+h//2), color, 2)
        cv2.putText(image, status, (x-w//2, y-h//2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    # Display counts
    cv2.putText(image, f"Occupied: {occupied}  Vacant: {vacant}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    return image