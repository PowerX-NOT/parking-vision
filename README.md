# Parking Slots Identification and Occupancy Tracking using Computer Vision

## Objective
Detect and track parking slot occupancy status using image or video input.

## Features

- Accepts images (.jpg/.png) and videos (.mp4/.avi)
- Detects parking slot locations
- Classifies slots as occupied or vacant
- Real-time visualization with colored boxes and counts

## Directory Structure

```
parking-vision/
├── data/                  # Sample images/videos for testing
├── models/                # Trained models (YOLOv5 weights, etc.)
├── notebooks/             # Prototyping and experimentation
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocess.py
│   ├── slot_detection.py
│   ├── occupancy.py
│   ├── visualization.py
│   ├── main.py
│   └── utils.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Quick Start

1. Install requirements:
    ```
    pip install -r requirements.txt
    ```

2. Run on an image:
    ```
    python src/main.py --image data/sample.jpg
    ```

3. Run on video:
    ```
    python src/main.py --video data/sample.mp4
    ```

## Notes

- Replace `"YOUR_WORKSPACE"` in `main.py` with your actual Roboflow workspace.
- For video inputs, frame extraction and display logic needs to be implemented in `main.py`.