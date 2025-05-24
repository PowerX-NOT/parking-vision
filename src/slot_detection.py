import torch

class YOLOv5VehicleDetector:
    def __init__(self, weights, device='cuda'):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, force_reload=False)
        self.device = device
        self.model.to(self.device).eval()
        # COCO classes for vehicles: car, truck, bus, motorcycle
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck in COCO

    def detect_vehicles(self, image):
        results = self.model(image)
        vehicles = []
        for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
            if int(cls) in self.vehicle_classes:
                x1, y1, x2, y2 = map(int, xyxy)
                vehicles.append({'bbox': (x1, y1, x2, y2), 'conf': float(conf), 'class': int(cls)})
        return vehicles
