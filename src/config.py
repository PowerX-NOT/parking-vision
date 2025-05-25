from pathlib import Path

# Path to YOLO OBB model
MODEL_PATH = str(Path(__file__).parent.parent / "models" / "yolo11l-obb.pt")

# Add any other configurable parameters here
# For example:
DEFAULT_IMAGE_SIZE = (1280, 720)