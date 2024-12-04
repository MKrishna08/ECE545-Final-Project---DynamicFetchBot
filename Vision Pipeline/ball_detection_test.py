import cv2
import numpy as np

class BallDetector:
    def __init__(self, lower_color_range, upper_color_range):
        self.lower_color = np.array(lower_color_range)
        self.upper_color = np.array(upper_color_range)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect_ball(self, frame):
        # Convert frame to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create a mask for the color range
        mask = cv2.inRange(hsv, self.lower_color, self.upper_color)
        
        # Apply morphological operations
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(max_contour) > 500:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(max_contour)
                center = (x + w // 2, y + h // 2)
                radius = max(w, h) // 2  # Approximate radius
                return center, (x, y, w, h, radius)

        # Return None when no ball is detected
        return None

    def draw_debug_info(self, frame, center, bounding_box):
        if center:
            x, y, w, h, _ = bounding_box
            cv2.circle(frame, center, 10, (0, 255, 0), -1)  # Mark center
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
