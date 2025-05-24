import cv2
import numpy as np

def classify_occupancy(image, slots, method="intensity"):
    status = []
    for slot in slots:
        x1, y1, x2, y2 = slot['bbox']
        slot_img = image[y1:y2, x1:x2]
        if method == "intensity":
            gray = cv2.cvtColor(slot_img, cv2.COLOR_BGR2GRAY)
            mean_intensity = np.mean(gray)
            # Threshold can be tuned
            occupied = mean_intensity < 100
        else:
            # For deep learning method, plug your classifier here
            occupied = False  # Placeholder
        status.append({'bbox': slot['bbox'], 'occupied': occupied})
    return status