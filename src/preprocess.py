import cv2

def resize_image(image, size):
    """
    Resize an image to the target size.
    """
    return cv2.resize(image, size)

def extract_frames(video_path, output_dir, frame_rate=1):
    """
    Extract frames from video at the given frame rate.
    """
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    frames = []
    while success:
        if count % frame_rate == 0:
            frames.append(image)
        success, image = vidcap.read()
        count += 1
    vidcap.release()
    return frames