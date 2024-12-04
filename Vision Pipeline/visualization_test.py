import cv2
import numpy as np

class Visualizer:
    def draw_ball_info(self, frame, center, bounding_box):
        if center and bounding_box:
            x, y, w, h, _ = bounding_box
            cv2.circle(frame, center, 10, (0, 255, 0), -1)  # Mark center
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Bounding box

    def draw_trajectory(self, frame, points, color=(255, 0, 0)):
        for i in range(1, len(points)):
            cv2.line(frame, points[i - 1], points[i], color, 2)

    def draw_landing_point(self, frame, mapped_x, mapped_y, square_size=2):
        # Assuming the frame represents the 2m x 2m square directly
        square_pixel_size = frame.shape[1]  # Assume the square fits the frame width
        scale = square_pixel_size / square_size
        pixel_x = int(mapped_x * scale)
        pixel_y = int(mapped_y * scale)

        cv2.circle(frame, (pixel_x, pixel_y), 15, (0, 0, 255), -1)
        cv2.putText(frame, f"Landing: ({mapped_x:.2f}, {mapped_y:.2f}) m",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    def compute_metrics(self, trail_points, predicted_trail_points):
        if len(trail_points) > 5 and len(predicted_trail_points) > 5:
            real_points = trail_points[-5:]
            pred_points = predicted_trail_points[-5:]
            mse = sum(((r[0] - p[0]) ** 2 + (r[1] - p[1]) ** 2) for r, p in zip(real_points, pred_points)) / len(real_points)
            accuracy = sum(1 for r, p in zip(real_points, pred_points) if abs(r[0] - p[0]) <= 10 and abs(r[1] - p[1]) <= 10) / len(real_points) * 100
            return mse, accuracy
        return None, None

    def display_metrics(self, frame, fps, mse, accuracy):
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if mse is not None:
            cv2.putText(frame, f"MSE: {mse:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if accuracy is not None:
            cv2.putText(frame, f"Accuracy: {accuracy:.2f}%", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
