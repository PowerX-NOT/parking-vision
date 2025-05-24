from ultralytics import YOLO

class YOLOv8VehicleDetector:
    def __init__(self, weights, device='cuda', conf=0.25, vehicle_keywords=None):
        self.model = YOLO(weights)
        self.device = device
        self.model.conf = conf  # Set confidence threshold
        if vehicle_keywords is None:
            self.vehicle_keywords = ["car", "truck", "bus", "motorcycle"]
        else:
            self.vehicle_keywords = [k.lower() for k in vehicle_keywords]

    def detect_vehicles(self, image):
        results = self.model.predict(source=image, device=0 if self.device=='cuda' else 'cpu', verbose=False)
        vehicles = []
        boxes = results[0].boxes
        names = self.model.names if hasattr(self.model, "names") else results[0].names
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
            for i in range(len(xyxy)):
                class_name = names[int(cls[i])].lower()
                if class_name in self.vehicle_keywords:
                    x1, y1, x2, y2 = map(int, xyxy[i])
                    vehicles.append({'bbox': (x1, y1, x2, y2), 'conf': float(conf[i]), 'class': class_name})
        return vehicles
