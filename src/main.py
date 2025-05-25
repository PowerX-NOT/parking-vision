import argparse
import os
import pickle
from ultralytics import YOLO
from src.slot_detection import detect_objects_polygons, check_slots_occupancy
from src.visualization import visualize_results
from src.preprocess import load_image, load_video_frames
from src.config import MODEL_PATH
from src.slot_selector import select_slots_on_image, SLOTS_FILE

def load_slots(slots_file=SLOTS_FILE):
    if os.path.exists(slots_file):
        with open(slots_file, "rb") as f:
            return pickle.load(f)
    else:
        print("No slots defined. Please run slot selection first.")
        return []

def process_image(image_path, model, slots):
    image = load_image(image_path)
    obb_polygons = detect_objects_polygons(image, model)
    statuses = check_slots_occupancy(slots, obb_polygons)
    result_img, dialog_img = visualize_results(image, slots, statuses, obb_polygons)
    import cv2
    cv2.imshow("Parking Slot Occupancy", result_img)
    cv2.imshow("Slot Summary", dialog_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_path, model, slots):
    import cv2
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        obb_polygons = detect_objects_polygons(frame, model)
        statuses = check_slots_occupancy(slots, obb_polygons)
        result_frame, dialog_frame = visualize_results(frame, slots, statuses, obb_polygons)
        cv2.imshow("Parking Slot Occupancy (Video)", result_frame)
        cv2.imshow("Slot Summary", dialog_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
    import gc
    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False

    def clear_memory():
        gc.collect()
        if has_torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 GPU memory cleared.")

    clear_memory()

    parser = argparse.ArgumentParser(description="Parking Slot Occupancy Detection (Image/Video)")
    parser.add_argument("--input", type=str, required=True, help="Path to image or video")
    parser.add_argument("--select_slots", action="store_true", help="Enter manual slot selection mode")
    args = parser.parse_args()

    if args.select_slots:
        select_slots_on_image(args.input)
        return

    slots = load_slots()
    if not slots:
        print("No slots loaded. Please mark slots first using --select_slots.")
        return

    model = YOLO(MODEL_PATH)

    if args.input.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        process_image(args.input, model, slots)
    elif args.input.lower().endswith(('.mp4', '.avi', '.mov')):
        process_video(args.input, model, slots)
    else:
        print("Error: Unsupported file format. Use .jpg/.png or .mp4/.avi")

    clear_memory()

if __name__ == "__main__":
    main()
