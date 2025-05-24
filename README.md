# Parking Slots Identification and Occupancy Tracking using Computer Vision

## 📌 Objective

Build a computer vision system to detect parking slots and track their occupancy status from images or video. The system provides real-time feedback to parking lot owners about current slot availability.

---

## ⚙️ Features

- Accepts both image (`.jpg`, `.png`) and video (`.mp4`, `.avi`) inputs.
- Detects parking slot locations using classical or deep learning methods (YOLOv5 supported).
- Classifies each slot as **occupied** or **vacant**.
- Provides real-time visualization with colored boxes (green for vacant, red for occupied) and slot counts.

---

## 🛠️ System Overview

1. **Preprocessing**
   - Resize input for consistent processing.
   - Extract frames from video if applicable.

2. **Parking Slot Detection**
   - Option 1: Classical techniques (edge detection, perspective correction)
   - Option 2: Deep learning-based detection (YOLOv5 model support)

3. **Occupancy Status Classification**
   - Detect if each slot is empty or has a car.
   - Methods: pixel intensity analysis or trained classifier.

4. **Visualization**
   - Draw color-coded rectangles for slot status.
   - Display live count of occupied and vacant slots.

---

## 🏗️ Directory Structure

```
parking-vision/
├── data/                  # Sample images/videos for testing
├── models/                # Trained models (YOLOv5 weights, etc.)
├── notebooks/             # Prototyping and experimentation
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration settings
│   ├── preprocess.py      # Preprocessing utilities (resize, frame extraction)
│   ├── slot_detection.py  # Parking slot detection logic
│   ├── occupancy.py       # Occupancy classification logic
│   ├── visualization.py   # Drawing & display logic
│   ├── main.py            # Entry point: CLI or GUI
│   └── utils.py           # Helper functions
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone <repo_url>
cd parking-vision
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

### 3. Place your YOLOv5 model weights

- Put the trained YOLOv5 weights (e.g., `best.pt`) into `models/`.

### 4. Run detection

**For image:**
```bash
python src/main.py --input data/sample.jpg --weights models/best.pt
```

**For video:**
```bash
python src/main.py --input data/sample.mp4 --weights models/best.pt
```

---

## 📦 Requirements

- Python 3.8+
- OpenCV
- PyTorch
- YOLOv5 (ultralytics/yolov5)
- numpy
- tqdm

See `requirements.txt` for more.

---

## 📝 Notes

- You can use your own images/videos by adding them to `data/`.
- You can prototype or visualize outputs in `notebooks/`.
- For classical methods, edit `src/slot_detection.py` accordingly.
- Visualization and result displays are handled by `src/visualization.py`.

---

## 💡 Acknowledgments

- [YOLOv5 by Ultralytics](https://github.com/ultralytics/yolov5)
- OpenCV community

---

Happy Parking! 🚗🅿️