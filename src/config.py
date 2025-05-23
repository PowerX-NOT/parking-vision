# Configuration settings for the parking vision project

IMAGE_SIZE = (640, 480)  # Width, Height
FRAME_SKIP = 1           # Process every frame (set >1 to skip frames)
MODEL_PATH = "models/slot_detector.pt"  # Path to trained model if used

# Thresholds for classical methods
EDGE_THRESHOLD = 50
OCCUPANCY_DIFF_THRESHOLD = 40

# Visualization settings
COLOR_VACANT = (0, 255, 0)   # Green
COLOR_OCCUPIED = (0, 0, 255) # Red