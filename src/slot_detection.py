import torch
import numpy as np
from src.config import YOLOV5_WEIGHTS, SLOT_CONF_THRESHOLD, SLOT_IOU_THRESHOLD

class YOLOv5SlotDetector:
    def __init__(self, weights=YOLOV5_WEIGHTS):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, force_reload=True)
        self.model.conf = SLOT_CONF_THRESHOLD
        self.model.iou = SLOT_IOU_THRESHOLD

    def detect_slots(self, image):
        results = self.model(image)
        slots = []
        for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
            x1, y1, x2, y2 = map(int, xyxy)
            slots.append({'bbox': (x1, y1, x2, y2), 'conf': float(conf), 'class': int(cls)})
        return slots