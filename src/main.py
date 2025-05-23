import argparse
import os
from preprocess import load_input, resize_frame
from slot_detection import YOLOv5SlotDetector
from occupancy import classify_occupancy
from visualization import draw_results, display_frame

def main(input_path):
    detector = YOLOv5SlotDetector()
    frames = load_input(input_path)
    is_image = os.path.splitext(input_path)[1].lower() in ['.jpg', '.png', '.jpeg']
    for frame in frames:
        frame = resize_frame(frame)
        slots = detector.detect_slots(frame)
        statuses = classify_occupancy(frame, slots)
        vis_frame = draw_results(frame.copy(), slots, statuses)
        if not display_frame(vis_frame, is_image=is_image):
            break
        if is_image:
            break
    import cv2
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parking Slots Identification and Occupancy Tracking")
    parser.add_argument("input_path", help="Path to image or video file")
    args = parser.parse_args()
    main(args.input_path)
