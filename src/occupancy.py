import cv2
import numpy as np
from config import OCCUPANCY_DIFF_THRESHOLD

def classify_occupancy(frame, slots, background=None):
    """
    Classify each slot as occupied or vacant.
    With background: compares pixel differences.
    Returns: list of 'occupied' or 'vacant'
    """
    statuses = []
    for (x, y, w, h) in slots:
        slot_roi = frame[y:y+h, x:x+w]
        if background is not None:
            bg_roi = background[y:y+h, x:x+w]
            diff = np.mean(np.abs(slot_roi.astype(float) - bg_roi.astype(float)))
            status = 'occupied' if diff > OCCUPANCY_DIFF_THRESHOLD else 'vacant'
        else:
            # Placeholder: treat all slots as vacant
            status = 'vacant'
        statuses.append(status)
    return statuses