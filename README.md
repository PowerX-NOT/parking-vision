# Parking Slots Identification and Occupancy Tracking using Computer Vision

## 📌 Objective

To build a computer vision system that detects parking slots and tracks their occupancy status using input from images or video. The system helps parking lot owners know the current availability status in real time.

---

## ⚙️ Features

- Accepts both image (`.jpg`, `.png`) and video (`.mp4`, `.avi`) as input
- Detects parking slot locations
- Classifies slots as **occupied** or **vacant**
- Displays real-time visualization with colored boxes and slot counts

---

## 🛠️ System Overview

### 1. Preprocessing

- Resize input for consistent processing
- Convert video into frames (if applicable)

### 2. Parking Slot Detection

- Use classical techniques (e.g., edge detection, perspective correction)
- Or deep learning-based detection (e.g., YOLOv8 OBB)

### 3. Occupancy Status Classification

- Detect if each slot is empty or has a car
- Use pixel intensity difference or trained models

### 4. Visualization

- Draw green (vacant) and red (occupied) rectangles
- Display count of occupied/vacant slots

---

## 📂 Directory Structure

```
parking-vision/
├── data/                  # Sample images/videos for testing
├── models/
│   └── yolo11l-obb.pt     # Pretrained YOLO OBB model
├── notebooks/             # Experimentation and prototyping
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

1. **Clone the repository**

   ```sh
   git clone https://github.com/your-username/parking-vision.git
   cd parking-vision
   ```

2. **Install dependencies**

   ```sh
   pip install -r requirements.txt
   ```

3. **Download the YOLO OBB model**

   Place `yolo11l-obb.pt` in the `models/` directory.

4. **Run the system**

   ```sh
   python src/main.py --input data/sample.jpg --output output
   ```

---

## 📝 Notes

- The `notebooks/` folder contains sample notebooks for prototyping and experimentation.
- For your own videos/images, place them in the `data/` directory.

---

## 📧 Contact

For any questions, open an issue or contact [your-email@example.com](mailto:your-email@example.com).