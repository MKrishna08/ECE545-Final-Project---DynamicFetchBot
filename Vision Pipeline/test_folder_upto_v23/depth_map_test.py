import cv2
import numpy as np

class DepthMap:
    def __init__(self, baseline=0.1, focal_length=700):
        if baseline <= 0 or focal_length <= 0:
            raise ValueError("Baseline and focal length must be positive values.")
        self.baseline = baseline
        self.focal_length = focal_length

    def compute_depth_map(self, left_frame, right_frame):
        # Validate frame shapes
        if left_frame.shape != right_frame.shape:
            raise ValueError("Left and right frames must have the same shape.")
        if len(left_frame.shape) != 3 or left_frame.shape[2] != 3:
            raise ValueError("Frames must be 3-channel color images.")
        
        # Convert frames to grayscale
        gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

        # Create stereo block matching object
        stereo = cv2.StereoBM_create(numDisparities=16 * 5, blockSize=15)

        # Compute disparity map
        disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
        if disparity.ndim != 2:
            raise ValueError("Disparity map must be a 2D array.")

        # Avoid division by zero
        disparity[disparity <= 0] = 0.1

        # Compute depth map
        depth_map = (self.focal_length * self.baseline) / disparity

        return depth_map

    def estimate_distance(self, depth_map, center):
        if depth_map is None or depth_map.size == 0:
            raise ValueError("Depth map is empty or invalid.")
        if not (isinstance(center, tuple) and len(center) == 2):
            raise ValueError("Center must be a tuple with two elements (x, y).")
        
        x, y = center
        if not (0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]):
            raise ValueError("Center coordinates are out of bounds.")

        distance = depth_map[int(y), int(x)]
        return distance

    def visualize_depth_map(self, depth_map, window_name="Depth Map"):
        if depth_map is None or depth_map.size == 0:
            raise ValueError("Depth map is empty or invalid.")

        normalized_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored_depth = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
        cv2.imshow(window_name, colored_depth)
