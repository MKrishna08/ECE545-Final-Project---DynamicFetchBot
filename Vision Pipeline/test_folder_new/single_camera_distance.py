import cv2
import numpy as np

class SingleCameraDistanceEstimator:
    def __init__(self, known_width, focal_length):
        """
        Initializes the single-camera distance estimator.

        Parameters:
        - known_width: The real-world width of the object (e.g., in meters).
        - focal_length: The focal length of the camera (in pixels), derived from calibration.
        """
        self.known_width = known_width
        self.focal_length = focal_length

    def estimate_distance(self, bounding_box_width):
        """
        Estimates the distance from the camera to the object.

        Parameters:
        - bounding_box_width: The width of the object's bounding box in the image (in pixels).

        Returns:
        - distance: The estimated distance from the object to the camera (in meters).
        """
        if bounding_box_width <= 0:
            return None
        distance = (self.known_width * self.focal_length) / bounding_box_width
        return distance

    def calibrate_focal_length(self, known_distance, bounding_box_width):
        """
        Calibrates the focal length of the camera based on a known distance and object size.

        Parameters:
        - known_distance: The real-world distance to the object (in meters).
        - bounding_box_width: The width of the object's bounding box in the image (in pixels).

        Returns:
        - focal_length: The calibrated focal length (in pixels).
        """
        self.focal_length = (bounding_box_width * known_distance) / self.known_width
        return self.focal_length
