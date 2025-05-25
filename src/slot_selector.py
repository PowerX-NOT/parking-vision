import cv2
import pickle
import os
import numpy as np

SLOTS_FILE = "slots.pkl"

class SlotSelector:
    def __init__(self, image, slots_file=SLOTS_FILE):
        self.image = image.copy()
        self.display = image.copy()
        self.slots_file = slots_file
        self.slots = []
        self.modification_mode = True  # Start with modification enabled
        self.drawing = False
        self.rect_start = None
        self.rect_end = None
        self.selected_idx = None
        self.load_slots()
        self.instructions = (
            "Drag left mouse: Add slot | Double-click inside: Delete slot | "
            "'m': Toggle mode | 's': Save | 'c': Clear | 'q': Quit"
        )

    def load_slots(self):
        if os.path.exists(self.slots_file):
            with open(self.slots_file, "rb") as f:
                self.slots = pickle.load(f)

    def save_slots(self):
        with open(self.slots_file, "wb") as f:
            pickle.dump(self.slots, f)
        print(f"Saved {len(self.slots)} slots to {self.slots_file}")

    def clear_slots(self):
        self.slots = []
        print("All slots cleared.")

    def draw_slots(self):
        self.display = self.image.copy()
        # Draw persistent slots
        for idx, (pt1, pt2) in enumerate(self.slots):
            color = (0, 255, 0) if idx != self.selected_idx else (0, 0, 255)
            cv2.rectangle(self.display, pt1, pt2, color, 2)
            cx, cy = (pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2
            cv2.putText(self.display, f"{idx+1}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        # Draw in-progress rectangle if drawing
        if self.drawing and self.rect_start and self.rect_end:
            cv2.rectangle(self.display, self.rect_start, self.rect_end, (255, 0, 0), 2)
        # Draw instructions
        cv2.rectangle(self.display, (0, 0), (self.display.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(self.display, self.instructions, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    def is_point_in_rect(self, point, rect):
        (x1, y1), (x2, y2) = rect
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])
        x, y = point
        return x_min <= x <= x_max and y_min <= y <= y_max

    def find_slot(self, point):
        for idx, rect in enumerate(self.slots):
            if self.is_point_in_rect(point, rect):
                return idx
        return None

    def mouse_callback(self, event, x, y, flags, param):
        if not self.modification_mode:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.rect_start = (x, y)
            self.rect_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.rect_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.rect_end = (x, y)
            # Only add if rectangle is of reasonable size
            if self.rect_start and self.rect_end and (abs(self.rect_start[0]-self.rect_end[0]) > 10 and abs(self.rect_start[1]-self.rect_end[1]) > 10):
                self.slots.append((self.rect_start, self.rect_end))
            self.drawing = False
            self.rect_start = None
            self.rect_end = None
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            idx = self.find_slot((x, y))
            if idx is not None:
                del self.slots[idx]
                print(f"Deleted slot {idx+1}")
        self.draw_slots()

    def run(self):
        cv2.namedWindow("Slot Selector")
        cv2.setMouseCallback("Slot Selector", self.mouse_callback)
        self.draw_slots()
        print(self.instructions)
        while True:
            self.draw_slots()
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
            self.draw_slots()
        cv2.destroyAllWindows()
        # Optionally save on exit
        self.save_slots()


def select_slots_on_image(image_path, slots_file=SLOTS_FILE):
    img = cv2.imread(image_path)
    selector = SlotSelector(img, slots_file)
    selector.run()
