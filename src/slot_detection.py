import torch

class YOLOv5SlotDetector:
    def __init__(self, weights="yolov5s.pt", device=None):
        self.model = torch.hub.load("ultralytics/yolov5", "custom", path=weights, force_reload=False)
        if device:
            self.model.to(device)
        self.model.eval()

    def detect_slots(self, frame):
        results = self.model(frame)
        slots = []
        for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
            if int(cls) in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                x1, y1, x2, y2 = map(int, xyxy)
                w, h = x2 - x1, y2 - y1
                slots.append((x1, y1, w, h))
        return slots
