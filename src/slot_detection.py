import cv2
import numpy as np

def detect_slots(frame):
    """
    Automatically detect parking slots using classical vision techniques.
    Returns a list of slot bounding boxes: [(x, y, w, h), ...]
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    # Line detection
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=70, maxLineGap=10)
    line_img = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (255,255,255), 2)
    
    # Find contours from the line image
    gray_lines = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    slots = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter out small or non-rectangular regions
        if 30 < w < 150 and 60 < h < 300 and 1.0 < h/w < 6.0:
            slots.append((x, y, w, h))
    return slots
