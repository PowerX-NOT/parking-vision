# Parking Slots Identification and Occupancy Tracking using Computer Vision

## Objective

A computer vision system that detects parking slots and tracks their occupancy status from images or video. The system helps parking lot owners know the current availability status in real time.

## Features

- Accepts both image (.jpg/.png) and video (.mp4/.avi) as input
- Detects parking slot locations (classical or deep learning methods)
- Classifies slots as occupied or vacant
- Displays real-time visualization with colored boxes and counts

## Project Structure

```plaintext
data/              # Sample images/videos for testing
models/            # Trained models, if any
notebooks/         # Experimentation
src/
  config.py        # Config/settings
  preprocess.py    # Preprocessing utilities
  slot_detection.py# Slot detection logic
  occupancy.py     # Occupancy classification
  visualization.py # Drawing & display
  main.py          # Entry point
  utils.py         # Helpers
requirements.txt
README.md
```

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the system

```bash
python src/main.py path/to/your/input.jpg
python src/main.py path/to/your/input.mp4
```

### 3. Customization

- Edit `src/config.py` for image size, thresholds, etc.
- Replace logic in `slot_detection.py` with a deep learning model for better accuracy.
- Add background reference for more robust occupancy detection.

## Requirements

- Python 3.8+
- OpenCV
- numpy

See `requirements.txt` for details.

## Notes

- The default pipeline uses simple classical methods. For production, integrate a trained deep learning model (e.g., YOLOv5) in `slot_detection.py`.
- For best results in occupancy, provide a background image or implement a trained classifier.

## License

MIT License