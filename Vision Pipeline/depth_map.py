import cv2
import numpy as np

class DepthMap:
    def __init__(self, baseline=0.1, focal_length=700):
        """
        Initializes the depth map module.
        
        Parameters:
        - baseline: Distance between the stereo cameras (in meters).
        - focal_length: Focal length of the cameras (in pixels).
        """
        self.baseline = baseline
        self.focal_length = focal_length

    def compute_depth_map(self, left_frame, right_frame):
        """
        Computes the depth map from stereo images.

        Parameters:
        - left_frame: Frame from the left camera.
        - right_frame: Frame from the right camera.

        Returns:
        - depth_map: The computed depth map.
        """
        # Convert frames to grayscale
        gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

        # Create stereo block matching object
        stereo = cv2.StereoBM_create(numDisparities=16 * 5, blockSize=15)

        # Compute disparity map
        disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

        # Avoid division by zero
        disparity[disparity <= 0] = 0.1

        # Compute depth map
        depth_map = (self.focal_length * self.baseline) / disparity

        return depth_map

    def estimate_distance(self, depth_map, center):
        """
        Estimates the distance of the object center from the camera.

        Parameters:
        - depth_map: The depth map.
        - center: The (x, y) coordinates of the object center.

        Returns:
        - distance: Estimated distance from the object to the camera (in meters).
        """
        x, y = center
        distance = depth_map[y, x]
        return distance

    def visualize_depth_map(self, depth_map, window_name="Depth Map"):
        """
        Displays the depth map.

        Parameters:
        - depth_map: The depth map to display.
        - window_name: The name of the display window.
        """
        normalized_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored_depth = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
        cv2.imshow(window_name, colored_depth)
