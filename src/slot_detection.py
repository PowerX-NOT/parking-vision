import cv2
import numpy as np

# Example: Define the slots as (x, y, w, h). You must fill in the correct coordinates for all slots in your image!
SLOT_RECTS = [
    (50, 40, 35, 75),    # slot 1 (top left)
    (50, 120, 35, 75),   # slot 2 (below slot 1)
    (50, 200, 35, 75),   # slot 3
    (50, 280, 35, 75),   # slot 4
    # ... Add all slots for your image for highest accuracy!
]

def detect_slots(frame):
    """
    Detect cars from aerial parking lot images using classical image processing.
    Returns a list of bounding boxes [(x, y, w, h), ...].
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    # Threshold to find bright regions (cars)
    _, binary = cv2.threshold(blurred, 170, 255, cv2.THRESH_BINARY)
    # Morphological operations to separate close cars
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Find contours
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    car_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect = w / (h + 1e-3)
        # Heuristic for "car-like" blobs: tune these numbers for your images!
        if 16 < w < 70 and 25 < h < 90 and 0.5 < aspect < 2.5 and area > 300:
            car_boxes.append((x, y, w, h))
    return car_boxes
