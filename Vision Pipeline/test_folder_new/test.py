import time
import cv2
from camera_feed import CameraFeed
from ball_detection import BallDetector
from motion_tracking import MotionTracker
from trajectory_prediction import TrajectoryPredictor
from visualization import Visualizer
from depth_map import DepthMap


# Initialize components
camera = CameraFeed(width=1200, height=720)
detector = BallDetector(lower_color_range=[40, 70, 70], upper_color_range=[80, 255, 255])
tracker = MotionTracker()
predictor = TrajectoryPredictor()
visualizer = Visualizer()
depth_estimator = DepthMap(baseline=0.1, focal_length=700)

# Initialize runtime variables
trail_points = []
predicted_trail_points = []
prev_time = time.time()

# Placeholder for stereo camera setup
left_camera = CameraFeed(width=1200, height=720)  # Left camera feed
right_camera = CameraFeed(width=1200, height=720)  # Right camera feed

try:
    while True:
        # Step 1: Capture frames from cameras
        frame = camera.get_frame()
        left_frame = left_camera.get_frame()
        right_frame = right_camera.get_frame()

        # Step 2: Detect the ball in the frame
        center, bounding_box = detector.detect_ball(frame)

        # Step 3: Kalman Filter prediction and correction
        predicted_center = tracker.predict()
        if center:
            tracker.correct(center)
            trail_points.append(center)
        predicted_trail_points.append(predicted_center)

        # Step 4: Update trajectory predictor
        if center:
            predictor.update_positions(center)

        predicted_landing = predictor.predict_landing(floor_y=720)  # Assuming the floor is at y=720

        # Step 5: Compute and visualize depth map
        depth_map = depth_estimator.compute_depth_map(left_frame, right_frame)
        if center:
            distance = depth_estimator.estimate_distance(depth_map, center)
            cv2.putText(frame, f"Distance: {distance:.2f} m", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        depth_estimator.visualize_depth_map(depth_map, window_name="Depth Map")

        # Step 6: Visualize real-time and predicted trajectories
        visualizer.draw_ball_info(frame, center, bounding_box)
        visualizer.draw_trajectory(frame, trail_points, color=(255, 0, 0))  # Blue for real trajectory
        visualizer.draw_trajectory(frame, predicted_trail_points, color=(0, 255, 255))  # Yellow for predicted trajectory

        # Step 7: Display metrics (FPS, MSE, Accuracy)
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        mse = None
        accuracy = None
        if len(trail_points) > 5 and len(predicted_trail_points) > 5:
            real_points = trail_points[-5:]
            pred_points = predicted_trail_points[-5:]
            mse = sum(((r[0] - p[0]) ** 2 + (r[1] - p[1]) ** 2) for r, p in zip(real_points, pred_points)) / len(real_points)
            accuracy = sum(1 for r, p in zip(real_points, pred_points) if abs(r[0] - p[0]) <= 10 and abs(r[1] - p[1]) <= 10) / len(real_points) * 100
        visualizer.display_metrics(frame, fps, mse, accuracy)

        # Step 8: Show the main camera feed with annotations
        camera.show_frame(frame, window_name="Ball Tracking System")

        # Step 9: Exit on key press
        if camera.exit_requested():
            break

        # Maintain trail point size
        trail_points = trail_points[-50:]  # Keep the last 50 points for clarity
        predicted_trail_points = predicted_trail_points[-50:]

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Release all resources
    camera.release()
    left_camera.release()
    right_camera.release()
    camera.close_windows()
