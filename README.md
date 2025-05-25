# 🚗 Parking Slots Identification and Occupancy Tracking using Computer Vision

## 📌 Objective

To build a computer vision system that detects parking slots and tracks their occupancy status using input from images or videos. This system helps parking lot owners monitor slot availability in real time.

---

## ⚙️ Features

- Supports both image (`.jpg`, `.png`) and video (`.mp4`, `.avi`) inputs  
- Detects parking slot locations  
- Classifies slots as **occupied** or **vacant**  
- Displays real-time visualization with colored boxes and slot counts  

---

## 🛠️ System Overview

### 1. Preprocessing

- Resize input for consistent processing  
- Convert video into frames (if applicable)

### 2. Parking Slot Detectiona

- Use classical techniques (e.g., edge detection, perspective correction)  
- Or deep learning-based detection (e.g., YOLOv8 OBB)

### 3. Occupancy Status Classification

- Detect vehicles using the object detection model (e.g., YOLOv8 OBB)
- If a detected vehicle overlaps with a marked slot (using IoU or bounding box intersection), mark the slot as **occupied**
- If a detected vehicle is not inside any slot, draw the vehicle in **blue**

**Color Codes:**
- Occupied slots: 🟥 Red  
- Vacant slots: 🟩 Green  
- Vehicles not inside any slot: 🔵 Blue

### 4. Visualization

- Draw **green** rectangles for vacant slots  
- Draw **red** rectangles for occupied slots  
- Draw vehicles:
  - **Red** if overlapping with a slot
  - **Blue** if not inside any slot  

---

## 📂 Directory Structure

```
parking-vision/
├── data/                  # Sample images/videos for testing
├── models/
│   └── yolo11x-obb.pt     # Pretrained YOLO OBB model
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

### 1. Clone the Repository

```bash
git clone https://github.com/PowerX-NOT/parking-vision.git
cd parking-vision
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🎯 Define Parking Slots (Manual Selection)

Run the following command:

```bash
python -m src.main --input data/sample.jpg --select_slots
```

### Manual Slot Selection & Editing UI (OpenCV-based):

- **Left-click and drag**: Define a rectangular parking slot  
- **Double-click inside a slot**: Delete that slot  
- Press **`s`**: Save current slots to `slots.pkl`  
- Press **`m`**: Toggle modification mode (add/remove slots)  
- Press **`c`**: Clear all marked slots  

> ✅ On startup, if `slots.pkl` exists, it will automatically be loaded.

---

## ▶️ Run the System

### For Image Input:

```bash
python -m src.main --input data/sample.jpg
```

### For Video Input:

```bash
python -m src.main --input data/sample.mp4
```

---

## 📌 License

This project is licensed under the [MIT License](LICENSE).

---
