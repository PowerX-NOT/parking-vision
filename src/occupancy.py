import numpy as np
import cv2

def classify_occupancy(image, slots):
    """
    Dummy occupancy classifier.
    For each slot polygon, check mean intensity or use a trained model.
    Returns a list of True (occupied) or False (vacant).
    """
    statuses = []
    for polygon in slots:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        pts = polygon.reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
        mean_val = cv2.mean(image, mask=mask)[:3]
        # Simple heuristic: darker areas = occupied (car), lighter = vacant
        occupied = np.mean(mean_val) < 100
        statuses.append(occupied)
    return statuses