import cv2
import pickle
import os

SLOTS_PICKLE = "slots.pkl"

class SlotSelector:
    def __init__(self, image_path, pickle_path=SLOTS_PICKLE):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Image not found at {image_path}")
        self.clone = self.image.copy()
        self.slots = []
        self.pickle_path = pickle_path
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.selected_slot_idx = None
        self.modifying = False

        # Load slots from pickle if exists
        if os.path.exists(self.pickle_path):
            with open(self.pickle_path, "rb") as f:
                self.slots = pickle.load(f)

    def select_slots(self):
        cv2.namedWindow("Select Slots")
        cv2.setMouseCallback("Select Slots", self.mouse_callback)
        print("Instructions:")
        print("  - Press 'm' to toggle modify mode (must be ON to add/remove slots)")
        print("  - In modify mode: Click and drag to draw a slot")
        print("  - Double-click inside a slot to delete it")
        print("  - Press 's' to save slots, 'ESC' to exit")
        while True:
            img = self.clone.copy()
            # Draw all slots
            for i, slot in enumerate(self.slots):
                color = (0,255,0)
                if self.selected_slot_idx == i:
                    color = (255,0,0)
                cv2.rectangle(img, (slot[0], slot[1]), (slot[2], slot[3]), color, 2)
            # Draw current rectangle if drawing
            if self.drawing and self.start_point and self.end_point:
                cv2.rectangle(img, self.start_point, self.end_point, (0,255,255), 2)
            cv2.imshow("Select Slots", img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC to exit
                break
            elif key == ord('s'):  # Save slots
                with open(self.pickle_path, "wb") as f:
                    pickle.dump(self.slots, f)
                print(f"Saved {len(self.slots)} slots to {self.pickle_path}")
            elif key == ord('m'):  # Toggle modify mode
                self.modifying = not self.modifying
                print("Modify mode:", "ON" if self.modifying else "OFF")
        cv2.destroyAllWindows()

    def mouse_callback(self, event, x, y, flags, param):
        if not self.modifying:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.end_point = (x, y)
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            slot = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            self.slots.append(slot)
            self.drawing = False
            self.start_point = None
            self.end_point = None
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            # On double left click, delete slot if inside one
            for i, slot in enumerate(self.slots):
                x1, y1, x2, y2 = slot
                if x1 <= x <= x2 and y1 <= y <= y2:
                    print(f"Deleting slot {i+1}")
                    del self.slots[i]
                    break
