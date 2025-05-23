import cv2
import numpy as np
from config import EDGE_THRESHOLD

def detect_slots(frame):
    """
    Detect parking slots in the frame.
    Returns a list of bounding boxes: [(x, y, w, h), ...]
    This is a stub using simple edge detection and contours.
    Replace with deep learning-based detection for better results.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, EDGE_THRESHOLD, EDGE_THRESHOLD*3)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    slots = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Heuristic: filter small or odd rectangles
        if w > 40 and h > 40 and 1 < w/h < 4:
            slots.append((x, y, w, h))
    return slots