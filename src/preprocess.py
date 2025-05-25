import cv2
from pathlib import Path

def load_image(image_path):
    """Loads and resizes an image."""
    image = cv2.imread(str(image_path))
    # Optionally resize here if needed
    return image

def load_video_frames(video_path):
    """Loads frames from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps, width, height