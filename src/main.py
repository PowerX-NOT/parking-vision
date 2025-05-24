import argparse
import cv2
import os

from src.config import YOLOV5_WEIGHTS
from src.preprocess import resize_image, extract_frames
from src.slot_detection import YOLOv5SlotDetector
from src.occupancy import classify_occupancy
from src.visualization import draw_slots, display_count
from src.utils import get_file_type

def process_image(image_path, slot_detector, occupancy_method):
    image = cv2.imread(image_path)
    image = resize_image(image)
    slots = slot_detector.detect_slots(image)
    slot_status = classify_occupancy(image, slots, method=occupancy_method)
    vis_img = draw_slots(image, slot_status)
    vis_img = display_count(vis_img, slot_status)
    cv2.imshow("Parking Slot Status", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_path, slot_detector, occupancy_method):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = resize_image(frame)
        slots = slot_detector.detect_slots(frame)
        slot_status = classify_occupancy(frame, slots, method=occupancy_method)
        vis_img = draw_slots(frame, slot_status)
        vis_img = display_count(vis_img, slot_status)
        cv2.imshow("Parking Slot Status", vis_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Parking Slots Identification and Occupancy Tracking")
    parser.add_argument("--input", type=str, required=True, help="Input image or video file path")
    parser.add_argument("--weights", type=str, default=YOLOV5_WEIGHTS, help="YOLOv5 weights path")
    parser.add_argument("--occupancy_method", type=str, default="intensity", help="Occupancy detection method")
    args = parser.parse_args()

    file_type = get_file_type(args.input)
    slot_detector = YOLOv5SlotDetector(weights=args.weights)

    if file_type == 'image':
        process_image(args.input, slot_detector, args.occupancy_method)
    elif file_type == 'video':
        process_video(args.input, slot_detector, args.occupancy_method)
    else:
        print("Unsupported input file type.")

if __name__ == "__main__":
    main()