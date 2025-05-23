import cv2
import numpy as np

def classify_occupancy(frame, slots):
    """
    Automatically classify each slot as occupied or vacant by checking color variance.
    Returns: list of 'occupied' or 'vacant'
    """
    statuses = []
    for (x, y, w, h) in slots:
        slot_roi = frame[y:y+h, x:x+w]
        # Convert ROI to grayscale for analysis
        gray_roi = cv2.cvtColor(slot_roi, cv2.COLOR_BGR2GRAY)
        # Heuristic: if the standard deviation is high, assume a car is present (more texture/edges)
        std = np.std(gray_roi)
        status = 'occupied' if std > 35 else 'vacant'
        statuses.append(status)
    return statuses
