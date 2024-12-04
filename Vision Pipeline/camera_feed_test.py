import cv2
import atexit
import time
import logging

logging.basicConfig(level=logging.INFO)

class CameraFeed:
    def __init__(self, camera_index=0, width=1200, height=720, fps=30):
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Frame rate control
        self.frame_delay = 1.0 / fps
        self.last_frame_time = time.time()
        
        if not self.cap.isOpened():
            logging.error("Error: Could not open camera.")
            raise Exception("Error: Could not open camera.")
        logging.info("Camera initialized successfully.")
        
        atexit.register(self.release)

    def get_frame(self):
        # Enforce frame rate
        now = time.time()
        while now - self.last_frame_time < self.frame_delay:
            now = time.time()
        self.last_frame_time = now
        
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Failed to grab frame")
        return frame

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
            logging.info("Camera released.")

    def show_frame(self, frame, window_name='Camera Feed'):
        cv2.imshow(window_name, frame)

    def exit_requested(self):
        return cv2.waitKey(1) & 0xFF == ord('q')

    def close_windows(self):
        cv2.destroyAllWindows()
