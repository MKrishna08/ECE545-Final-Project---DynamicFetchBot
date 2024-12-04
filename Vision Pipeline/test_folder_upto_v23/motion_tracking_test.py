import cv2
import numpy as np

class MotionTracker:
    def __init__(self, process_noise=0.03, measurement_noise=0.1):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise

    def predict(self):
        predicted = self.kalman.predict()
        position = (int(predicted[0]), int(predicted[1]))
        velocity = (predicted[2], predicted[3])  # Velocity in x and y
        return position, velocity

    def correct(self, measured_center, confidence=1.0):
        measurement = np.array([[np.float32(measured_center[0])], [np.float32(measured_center[1])]])
        self.kalman.correct(measurement * confidence)

    def handle_lost_tracking(self):
        # Predict the next position even if there is no new measurement
        return self.predict()

    def save_state(self):
        return self.kalman.statePost.copy()

    def restore_state(self, state):
        self.kalman.statePost = state

    def debug_state(self):
        print("Kalman State:")
        print(f"Position: ({self.kalman.statePost[0]}, {self.kalman.statePost[1]})")
        print(f"Velocity: ({self.kalman.statePost[2]}, {self.kalman.statePost[3]})")
