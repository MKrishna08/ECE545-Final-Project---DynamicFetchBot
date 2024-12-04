import time
import cv2
from camera_feed_test import CameraFeed
from ball_detection_test import BallDetector
from motion_tracking_test import MotionTracker
from trajectory_prediction_test import TrajectoryPredictor
from visualization_test import Visualizer
from depth_map import DepthMap
from single_camera_distance_test import SingleCameraDistanceEstimator

# Initialize components
camera = CameraFeed(width=1200, height=720)
detector = BallDetector(lower_color_range=[40, 70, 70], upper_color_range=[80, 255, 255])
tracker = MotionTracker()
predictor = TrajectoryPredictor()
visualizer = Visualizer()
depth_estimator = DepthMap(baseline=0.1, focal_length=700)
distance_estimator = SingleCameraDistanceEstimator(known_width=0.2, focal_length=800)

# Initialize runtime variables
trail_points = []
predicted_trail_points = []
prev_time = time.time()

try:
    while True:
        # Step 1: Capture frames from the camera
        frame = camera.get_frame()

        # Step 2: Detect the ball in the frame
        result = detector.detect_ball(frame)  # Safely get the result
        if result is not None:
            center, bounding_box = result
        else:
            center, bounding_box = None, None

        # Step 3: Kalman Filter prediction and correction
        predicted_center, _ = tracker.predict()
        if center is not None:
            tracker.correct(center)
            trail_points.append(center)
        else:
            # Handle no detection case gracefully
            center = tracker.handle_lost_tracking()
        predicted_trail_points.append(predicted_center)

        # Step 4: Update trajectory predictor
        if center:
            predictor.update_positions(center)

        predicted_landing = predictor.predict_landing(floor_y=720)  # Assuming floor is at y=720
        if predicted_landing:
            mapped_x, mapped_y = predictor.map_to_square(predicted_landing, square_size=2)
            with open("landing_positions.txt", "a") as file:
                file.write(f"{mapped_x:.2f}, {mapped_y:.2f}\n")
            visualizer.draw_landing_point(frame, mapped_x, mapped_y, square_size=2)

        # Step 5: Visualize real-time and predicted trajectories
        visualizer.draw_ball_info(frame, center, bounding_box)
        visualizer.draw_trajectory(frame, trail_points, color=(255, 0, 0))  # Blue for real trajectory
        visualizer.draw_trajectory(frame, predicted_trail_points, color=(0, 255, 255))  # Yellow for predicted trajectory

        # Step 6: Display FPS and metrics
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        mse, accuracy = visualizer.compute_metrics(trail_points, predicted_trail_points)
        visualizer.display_metrics(frame, fps, mse, accuracy)

        # Step 7: Single Camera Distance Estimation
        if bounding_box is not None:
            _, _, box_width, _ = bounding_box
            single_camera_distance = distance_estimator.estimate_distance(box_width)
            if single_camera_distance:
                cv2.putText(frame, f"Distance (1-Cam): {single_camera_distance:.2f} m",
                            (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Step 8: Show the camera feed with annotations
        camera.show_frame(frame, window_name="Ball Tracking System w/ Single Camera")

        # Step 9: Exit on key press
        if camera.exit_requested():
            break

        # Maintain trail point size
        trail_points = trail_points[-50:]  # Keep the last 50 points
        predicted_trail_points = predicted_trail_points[-50:]

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Release all resources
    camera.release()
    camera.close_windows()