import os

# Default configuration for the project

DATA_DIR = os.getenv("PV_DATA_DIR", "../data")
MODELS_DIR = os.getenv("PV_MODELS_DIR", "../models")
FRAME_SIZE = (640, 480)
YOLOV5_WEIGHTS = os.getenv("PV_YOLOV5_WEIGHTS", os.path.join(MODELS_DIR, "best.pt"))

# Slot detection
SLOT_CONF_THRESHOLD = 0.4
SLOT_IOU_THRESHOLD = 0.45

# Occupancy
OCCUPANCY_METHOD = os.getenv("PV_OCCUPANCY_METHOD", "yolov5")  # or 'classical'