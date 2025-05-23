import argparse
import cv2
from preprocess import load_input, resize_frame
from slot_detection import detect_slots
from occupancy import classify_occupancy
from visualization import draw_results, display_frame

def main(input_path):
    frames = load_input(input_path)
    for frame in frames:
        frame = resize_frame(frame)
        slots = detect_slots(frame)
        statuses = classify_occupancy(frame, slots)
        vis_frame = draw_results(frame.copy(), slots, statuses)
        if not display_frame(vis_frame):
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parking Slots Identification and Occupancy Tracking")
    parser.add_argument("input_path", help="Path to image or video file")
    args = parser.parse_args()
    main(args.input_path)
