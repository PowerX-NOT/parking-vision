# Configuration settings for the Parking Vision project

IMAGE_SIZE = (640, 480)
VIDEO_FRAME_RATE = 25  # Frames per second for processing
MODEL_CONFIDENCE = 40  # Roboflow/YOLOv5 confidence threshold
MODEL_OVERLAP = 30     # Roboflow/YOLOv5 overlap threshold

# Colors for visualization
COLOR_VACANT = (0, 255, 0)   # Green
COLOR_OCCUPIED = (0, 0, 255) # Red