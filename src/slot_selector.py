import cv2
import pickle
import os
import numpy as np
from roboflow import Roboflow

SLOTS_FILE = "slots.pkl"

class SlotSelector:
    def __init__(self, image, slots_file=SLOTS_FILE):
        self.image = image.copy()
        self.display = image.copy()
        self.slots_file = slots_file
        self.slots = []  # Stores all saved slots
        self.all_detections = []  # Stores all detections from Roboflow
        self.modification_mode = True
        self.drawing = False
        self.rect_start = None
        self.rect_end = None
        self.selected_idx = None
        self.load_slots()
        
        # Initialize Roboflow
        self.rf = Roboflow(api_key="4Ffffb9rznBnFJyZbsHv")
        self.project = self.rf.workspace().project("deteksiparkirkosong")
        self.model = self.project.version(6).model
        
        self.instructions = (
            "Left click: Select slot | Drag: Add new slot | Double-click: Delete slot | "
            "'m': Toggle mode | 's': Save | 'c': Clear | 'd': Detect | 'r': Save all red slots | 'q': Quit"
        )

    def load_slots(self):
        if os.path.exists(self.slots_file):
            with open(self.slots_file, "rb") as f:
                self.slots = pickle.load(f)
                # Ensure loaded slots are in the correct format
                self.slots = [((int(x1), int(y1)), (int(x2), int(y2))) for ((x1, y1), (x2, y2)) in self.slots]

    def save_slots(self):
        with open(self.slots_file, "wb") as f:
            pickle.dump(self.slots, f)
        print(f"Saved {len(self.slots)} slots to {self.slots_file}")

    def clear_slots(self):
        self.slots = []
        self.all_detections = []
        print("All slots cleared.")

    def detect_parking_spaces(self):
        temp_path = "temp_image.jpg"
        cv2.imwrite(temp_path, self.image)
        
        try:
            predictions = self.model.predict(temp_path, confidence=40, overlap=30).json()['predictions']
            self.all_detections = predictions
            
            # Clear existing slots and add only free spaces
            self.slots = []
            for pred in predictions:
                if pred["class"].lower() == "free":
                    x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
                    x1, y1 = int(x - w / 2), int(y - h / 2)
                    x2, y2 = int(x + w / 2), int(y + h / 2)
                    self.slots.append(((x1, y1), (x2, y2)))
            
            print(f"Detected {len(predictions)} parking spaces ({len(self.slots)} free)")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def save_all_red_slots(self):
        """Convert all red (occupied) slots to yellow (saved) slots"""
        count = 0
        for pred in self.all_detections:
            if pred["class"].lower() == "car":
                x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
                x1, y1 = int(x - w / 2), int(y - h / 2)
                x2, y2 = int(x + w / 2), int(y + h / 2)
                new_slot = ((x1, y1), (x2, y2))
                
                if not any(self.is_same_rect(new_slot, s) for s in self.slots):
                    self.slots.append(new_slot)
                    count += 1
        
        print(f"Converted {count} occupied slots to saved slots")
        self.draw_slots()

    def draw_slots(self):
        self.display = self.image.copy()
        
        # Draw all saved slots in yellow
        for idx, (pt1, pt2) in enumerate(self.slots):
            color = (0, 255, 255)  # Yellow for saved slots
            cv2.rectangle(self.display, pt1, pt2, color, 2)
            cx, cy = (pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2
            cv2.putText(self.display, f"{idx+1}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw all detections (free and occupied)
        for pred in self.all_detections:
            label = pred["class"].lower()
            x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)
            
            # Skip if this detection is already saved
            if any(self.is_same_rect(((x1, y1), (x2, y2)), s) for s in self.slots):
                continue
                
            if label == "free":
                color = (0, 255, 0)  # Green for free
                cv2.rectangle(self.display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(self.display, "Free", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            elif label == "car":
                color = (0, 0, 255)  # Red for occupied
                cv2.rectangle(self.display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(self.display, "", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Highlight selected slot
        if self.selected_idx is not None and self.selected_idx < len(self.slots):
            pt1, pt2 = self.slots[self.selected_idx]
            cv2.rectangle(self.display, pt1, pt2, (255, 0, 255), 3)  # Purple for selected
        
        # Draw in-progress rectangle if drawing
        if self.drawing and self.rect_start and self.rect_end:
            cv2.rectangle(self.display, self.rect_start, self.rect_end, (255, 0, 0), 2)
        
        # Draw instructions
        cv2.rectangle(self.display, (0, 0), (self.display.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(self.display, self.instructions, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    def is_same_rect(self, rect1, rect2, threshold=10):
        """Check if two rectangles are approximately the same"""
        (x1a, y1a), (x2a, y2a) = rect1
        (x1b, y1b), (x2b, y2b) = rect2
        return (abs(x1a - x1b) < threshold and abs(y1a - y1b) < threshold and
               abs(x2a - x2b) < threshold and abs(y2a - y2b) < threshold)

    def find_slot_at_point(self, point):
        """Find both saved slots and detections at a point"""
        # First check saved slots
        for idx, rect in enumerate(self.slots):
            if self.is_point_in_rect(point, rect):
                return ('saved', idx)
        
        # Then check detections
        for idx, pred in enumerate(self.all_detections):
            x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)
            if self.is_point_in_rect(point, ((x1, y1), (x2, y2))):
                return ('detection', idx)
        
        return (None, None)

    def is_point_in_rect(self, point, rect):
        (x1, y1), (x2, y2) = rect
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])
        x, y = point
        return x_min <= x <= x_max and y_min <= y <= y_max

    def mouse_callback(self, event, x, y, flags, param):
        if not self.modification_mode:
            return
            
        point = (x, y)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicking on an existing slot or detection
            slot_type, idx = self.find_slot_at_point(point)
            
            if slot_type == 'saved':
                self.selected_idx = idx
            elif slot_type == 'detection':
                # Convert this detection to a saved slot
                pred = self.all_detections[idx]
                x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
                x1, y1 = int(x - w / 2), int(y - h / 2)
                x2, y2 = int(x + w / 2), int(y + h / 2)
                new_slot = ((x1, y1), (x2, y2))
                
                if not any(self.is_same_rect(new_slot, s) for s in self.slots):
                    self.slots.append(new_slot)
                    self.selected_idx = len(self.slots) - 1
                    print(f"Added slot from detection (previously {pred['class']})")
            
            # Start drawing new rectangle if not clicking on existing slot
            if slot_type is None:
                self.drawing = True
                self.rect_start = (x, y)
                self.rect_end = (x, y)
                
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.rect_end = (x, y)
            self.draw_slots()  # Update display while dragging
            
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.rect_end = (x, y)
            if self.rect_start and self.rect_end and (
                abs(self.rect_start[0]-self.rect_end[0]) > 10 and 
                abs(self.rect_start[1]-self.rect_end[1]) > 10
            ):
                new_slot = (self.rect_start, self.rect_end)
                if not any(self.is_same_rect(new_slot, s) for s in self.slots):
                    self.slots.append(new_slot)
                    self.selected_idx = len(self.slots) - 1
                    print("Added new manual slot")
            self.drawing = False
            self.rect_start = None
            self.rect_end = None
            
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            slot_type, idx = self.find_slot_at_point(point)
            if slot_type == 'saved':
                del self.slots[idx]
                if self.selected_idx == idx:
                    self.selected_idx = None
                elif self.selected_idx is not None and self.selected_idx > idx:
                    self.selected_idx -= 1
                print(f"Deleted slot {idx+1}")
                
        self.draw_slots()

    def run(self):
        cv2.namedWindow("Slot Selector")
        cv2.setMouseCallback("Slot Selector", self.mouse_callback)
        self.draw_slots()
        print(self.instructions)
        while True:
            cv2.imshow("Slot Selector", self.display)
            key = cv2.waitKey(20) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                self.modification_mode = not self.modification_mode
                print(f"Modification mode {'ON' if self.modification_mode else 'OFF'}")
            elif key == ord('s'):
                self.save_slots()
            elif key == ord('c'):
                self.clear_slots()
            elif key == ord('d'):
                self.detect_parking_spaces()
            elif key == ord('r'):
                self.save_all_red_slots()
            self.draw_slots()
        cv2.destroyAllWindows()
        self.save_slots()


def select_slots_on_image(image_path, slots_file=SLOTS_FILE):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    selector = SlotSelector(img, slots_file)
    selector.run()
