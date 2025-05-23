import cv2
import os

from config import IMAGE_SIZE

def load_input(input_path):
    """Load images or extract frames from video"""
    if input_path.lower().endswith(('.jpg', '.png')):
        frame = cv2.imread(input_path)
        return [frame]
    elif input_path.lower().endswith(('.mp4', '.avi')):
        return extract_frames(input_path)
    else:
        raise ValueError("Unsupported file type")

def extract_frames(video_path, skip=1):
    """Extract frames from video"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % skip == 0:
            frames.append(frame)
        frame_id += 1
    cap.release()
    return frames

def resize_frame(frame):
    return cv2.resize(frame, IMAGE_SIZE)