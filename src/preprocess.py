import cv2
import os

def resize_image(img, size=(640, 480)):
    return cv2.resize(img, size)

def extract_frames(video_path, output_dir, frame_skip=1):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    saved_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_skip == 0:
            fname = os.path.join(output_dir, f"frame_{frame_idx:05d}.jpg")
            cv2.imwrite(fname, frame)
            saved_count += 1
        frame_idx += 1
    cap.release()
    return saved_count