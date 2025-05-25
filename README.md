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

### 1. Parking Slot Detection

- Utilizes a deep learning-based object detection model **YOLOv11x OBB** to identify and map parking slots.

### 2. Occupancy Status Classification

- Vehicles are detected using the same object detection model.
- For each vehicle:
  - If it overlaps with a marked parking slot (based on IoU or bounding box intersection), the slot is marked as **occupied**.
  - If the vehicle does **not** fall within any marked slot, it is highlighted in **blue**.

#### 🎨 Color Code:

- 🟥 **Red** – Occupied Slots  
- 🟩 **Green** – Vacant Slots  
- 🔵 **Blue** – Vehicles not inside any marked slot

### 3. Output Summary

A summary file (`.csv`) is generated with the following details:

| Total Slots | Occupied Slots | Available Slots |
|-------------|----------------|-----------------|
| 400         | 136            | 264             |

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

## 🎥 Demo

![Demo Screenshot](https://github.com/PowerX-NOT/parking-vision/blob/242d3c729dc007ea4f569d985664f2af94de6bf8/demo/demo.png)

![Demo GIF](https://github.com/PowerX-NOT/parking-vision/blob/242d3c729dc007ea4f569d985664f2af94de6bf8/demo/demo.gif)

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
