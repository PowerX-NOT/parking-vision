import argparse
from pathlib import Path
import gc

try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False

from ultralytics import YOLO
from src.slot_detection import detect_slots
from src.occupancy import classify_occupancy
from src.visualization import visualize_results
from src.preprocess import load_image, load_video_frames
from src.config import MODEL_PATH

def clear_memory():
    """Clear Python and CUDA memory."""
    gc.collect()
    if has_torch and torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 GPU memory cleared.")

def process_image(image_path, model):
    """Process a single image for parking slot detection and occupancy (display only)."""
    image = load_image(image_path)
    slots = detect_slots(image, model)
    statuses = classify_occupancy(image, slots)
    result_img = visualize_results(image, slots, statuses)

    # Display only, do not save
    from cv2 import imshow, waitKey, destroyAllWindows
    imshow("Parking Slot Occupancy", result_img)
    waitKey(0)
    destroyAllWindows()

def process_video(video_path, model):
    """
    Process a video for parking slot detection and occupancy (display only).
    """
    frames, fps, width, height = load_video_frames(video_path)
    from cv2 import imshow, waitKey, destroyAllWindows

    for frame in frames:
        slots = detect_slots(frame, model)
        statuses = classify_occupancy(frame, slots)
        result_frame = visualize_results(frame, slots, statuses)
        imshow("Parking Slot Occupancy (Video)", result_frame)
        if waitKey(1) & 0xFF == ord('q'):
            break
    destroyAllWindows()

def main():
    clear_memory()  # Clear memory before execution

    parser = argparse.ArgumentParser(description="Parking Slot Occupancy Detection (Image/Video)")
    parser.add_argument("--input", type=str, required=True, help="Path to image or video")
    args = parser.parse_args()

    # Load model
    model = YOLO(MODEL_PATH)

    # Check if input is image or video
    if args.input.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        process_image(args.input, model)
    elif args.input.lower().endswith(('.mp4', '.avi', '.mov')):
        process_video(args.input, model)
    else:
        print("Error: Unsupported file format. Use .jpg/.png or .mp4/.avi")

    clear_memory()  # Clear memory after execution

if __name__ == "__main__":
    main()
