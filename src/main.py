import argparse
import cv2
from preprocess import load_input, resize_frame
from slot_detection import YOLOv5SlotDetector
from occupancy import classify_occupancy
from visualization import draw_results, display_frame

def main(input_path, weights="yolov5s.pt"):
    detector = YOLOv5SlotDetector(weights)
    frames = load_input(input_path)
    for frame in frames:
        frame = resize_frame(frame)
        slots = detector.detect_slots(frame)
        statuses = classify_occupancy(frame, slots)
        vis_frame = draw_results(frame.copy(), slots, statuses)
        if not display_frame(vis_frame):
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parking Slots Identification and Occupancy Tracking (YOLOv5)")
    parser.add_argument("input_path", help="Path to image or video file")
    parser.add_argument("--weights", default="yolov5s.pt", help="YOLOv5 model weights path")
    args = parser.parse_args()
    main(args.input_path, args.weights)
