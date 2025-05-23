import torch
import numpy as np

class YOLOv5SlotDetector:
    def __init__(self, weights="yolov5s.pt", device=None):
        self.model = torch.hub.load("ultralytics/yolov5", "custom", path=weights, force_reload=False)
        if device:
            self.model.to(device)
        self.model.eval()

    def detect_slots(self, frame):
        """
        Detect cars as 'slots' (occupied) using YOLOv5.
        Returns a list of bounding boxes (x, y, w, h) for each detected car.
        """
        results = self.model(frame)
        slots = []
        for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
            # COCO car class is 2, truck is 7, bus is 5, motorcycle is 3, etc.
            if int(cls) in [2, 3, 5, 7]:
                x1, y1, x2, y2 = map(int, xyxy)
                w, h = x2 - x1, y2 - y1
                slots.append((x1, y1, w, h))
        return slots
