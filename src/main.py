import argparse
import os
import cv2
from roboflow import Roboflow

def is_image_file(file_path):
    return os.path.splitext(file_path)[1].lower() in ['.jpg', '.jpeg', '.png']

def is_video_file(file_path):
    return os.path.splitext(file_path)[1].lower() in ['.mp4', '.avi', '.mov', '.mkv']

def draw_boxes_from_roboflow(image_path, result):
    image = cv2.imread(image_path)
    for pred in result["predictions"]:
        x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
        class_name = pred["class"]
        top_left = (x - w // 2, y - h // 2)
        bottom_right = (x + w // 2, y + h // 2)
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(image, class_name, (top_left[0], top_left[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imshow("Detections", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_image(model, input_path):
    result = model.predict(input_path, confidence=40, overlap=30).json()
    draw_boxes_from_roboflow(input_path, result)

def process_video(model, input_path):
    cap = cv2.VideoCapture(input_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        temp_path = "temp_frame.jpg"
        cv2.imwrite(temp_path, frame)
        result = model.predict(temp_path, confidence=40, overlap=30).json()
        image = cv2.imread(temp_path)
        for pred in result["predictions"]:
            x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
            class_name = pred["class"]
            top_left = (x - w // 2, y - h // 2)
            bottom_right = (x + w // 2, y + h // 2)
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(image, class_name, (top_left[0], top_left[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow("Detections", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Object Detection for Parking Vision")
    parser.add_argument('--input', type=str, required=True, help='Path to input image or video')
    args = parser.parse_args()
    input_path = args.input

    rf = Roboflow(api_key="4Ffffb9rznBnFJyZbsHv")
    project = rf.workspace().project("car-parking-log8l")
    model = project.version(2).model

    if is_image_file(input_path):
        process_image(model, input_path)
    elif is_video_file(input_path):
        process_video(model, input_path)
    else:
        print("Unsupported file type. Please provide an image or video.")

if __name__ == "__main__":
    main()
