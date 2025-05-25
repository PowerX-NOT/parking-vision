import argparse
from pathlib import Path
from ultralytics import YOLO
from src.slot_detection import detect_slots
from src.occupancy import classify_occupancy
from src.visualization import visualize_results
from src.preprocess import load_image, load_video_frames
from src.config import MODEL_PATH

def process_image(image_path, model, output_dir="output"):
    """Process a single image for parking slot detection and occupancy."""
    image = load_image(image_path)
    slots = detect_slots(image, model)
    statuses = classify_occupancy(image, slots)
    result_img = visualize_results(image, slots, statuses)

    # Save & display
    output_path = Path(output_dir) / f"result_{Path(image_path).name}"
    Path(output_dir).mkdir(exist_ok=True)
    from cv2 import imwrite, imshow, waitKey, destroyAllWindows
    imwrite(str(output_path), result_img)
    imshow("Parking Slot Occupancy", result_img)
    waitKey(0)
    destroyAllWindows()

def process_video(video_path, model, output_dir="output"):
    """Process a video for parking slot detection and occupancy."""
    frames, fps, width, height = load_video_frames(video_path)
    Path(output_dir).mkdir(exist_ok=True)
    output_path = Path(output_dir) / f"result_{Path(video_path).name}"
    from cv2 import VideoWriter, VideoWriter_fourcc, imshow, waitKey, destroyAllWindows

    out = VideoWriter(str(output_path), VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in frames:
        slots = detect_slots(frame, model)
        statuses = classify_occupancy(frame, slots)
        result_frame = visualize_results(frame, slots, statuses)
        out.write(result_frame)
        imshow("Parking Slot Occupancy (Video)", result_frame)
        if waitKey(1) & 0xFF == ord('q'):
            break
    out.release()
    destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Parking Slot Occupancy Detection (Image/Video)")
    parser.add_argument("--input", type=str, required=True, help="Path to image or video")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    args = parser.parse_args()

    # Load model
    model = YOLO(MODEL_PATH)

    # Check if input is image or video
    if args.input.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        process_image(args.input, model, args.output)
    elif args.input.lower().endswith(('.mp4', '.avi', '.mov')):
        process_video(args.input, model, args.output)
    else:
        print("Error: Unsupported file format. Use .jpg/.png or .mp4/.avi")

if __name__ == "__main__":
    main()