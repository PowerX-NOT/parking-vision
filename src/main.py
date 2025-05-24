import argparse
import cv2
import torch
import gc
import pickle
import os

from src.slot_selector import SlotSelector, SLOTS_PICKLE
from src.slot_detection import YOLOv5VehicleDetector
from src.occupancy import classify_occupancy
from src.visualization import draw_slots, display_count
from src.utils import get_file_type

def load_slots(pickle_path):
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            return pickle.load(f)
    else:
        return []

def process_image(image_path, vehicle_detector, slots):
    image = cv2.imread(image_path)
    vehicles = vehicle_detector.detect_vehicles(image)
    slot_status = classify_occupancy(slots, vehicles)
    vis_img = draw_slots(image, slot_status)
    vis_img = display_count(vis_img, slot_status)
    cv2.imshow("Parking Slot Status", vis_img)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            break
    cv2.destroyAllWindows()

def process_video(video_path, vehicle_detector, slots):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        vehicles = vehicle_detector.detect_vehicles(frame)
        slot_status = classify_occupancy(slots, vehicles)
        vis_img = draw_slots(frame, slot_status)
        vis_img = display_count(vis_img, slot_status)
        cv2.imshow("Parking Slot Status", vis_img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Parking Slots Identification and Occupancy Tracking")
    parser.add_argument("--input", type=str, required=True, help="Image or video file path")
    parser.add_argument("--weights", type=str, required=True, help="YOLOv5 weights path")
    parser.add_argument("--select_slots", action='store_true', help="Enter manual slot selection/editor mode")
    args = parser.parse_args()

    # Clear GPU memory before
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    if args.select_slots:
        selector = SlotSelector(args.input)
        selector.select_slots()
        return

    slots = load_slots(SLOTS_PICKLE)
    if not slots:
        print("No slots found. Run with --select_slots first.")
        return

    vehicle_detector = YOLOv5VehicleDetector(args.weights)
    file_type = get_file_type(args.input)

    try:
        if file_type == 'image':
            process_image(args.input, vehicle_detector, slots)
        elif file_type == 'video':
            process_video(args.input, vehicle_detector, slots)
        else:
            print("Unsupported input file type.")
    finally:
        del vehicle_detector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("🧹 GPU memory cleared.")

if __name__ == "__main__":
    main()
