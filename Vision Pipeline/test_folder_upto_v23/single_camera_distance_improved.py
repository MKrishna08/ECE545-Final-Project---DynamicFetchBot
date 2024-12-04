import cv2
import numpy as np

class SingleCameraDistanceEstimator:
    def __init__(self, known_width, focal_length=None, verbose=False):
        """
        Initializes the single-camera distance estimator.

        Parameters:
        - known_width: The real-world width of the object (e.g., in meters).
        - focal_length: The focal length of the camera (in pixels). If None, it must be calibrated later.
        - verbose: If True, prints debug information during distance estimation.
        """
        self.known_width = known_width
        self.focal_length = focal_length
        self.verbose = verbose

    def estimate_distance(self, bounding_box_width):
        """
        Estimates the distance from the camera to the object.

        Parameters:
        - bounding_box_width: The width of the object's bounding box in the image (in pixels).

        Returns:
        - distance: The estimated distance from the object to the camera (in meters), or None if invalid.
        """
        if bounding_box_width <= 0:
            if self.verbose:
                print("Invalid bounding box width. Cannot estimate distance.")
            return None
        
        if self.focal_length is None:
            if self.verbose:
                print("Focal length is not calibrated. Cannot estimate distance.")
            return None

        # Estimate distance using the pinhole camera model
        distance = (self.known_width * self.focal_length) / bounding_box_width

        if self.verbose:
            print(f"Bounding Box Width: {bounding_box_width} pixels")
            print(f"Estimated Distance: {distance:.2f} meters")

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
        if known_distance <= 0 or bounding_box_width <= 0:
            raise ValueError("Known distance and bounding box width must be positive values.")

        self.focal_length = (bounding_box_width * known_distance) / self.known_width

        if self.verbose:
            print(f"Calibrated Focal Length: {self.focal_length:.2f} pixels")

        return self.focal_length

    def estimate_multiple_objects(self, bounding_boxes):
        """
        Estimates distances for multiple objects.

        Parameters:
        - bounding_boxes: A list of bounding box widths (in pixels).

        Returns:
        - distances: A list of estimated distances (in meters) for each object.
        """
        distances = []
        for box_width in bounding_boxes:
            distance = self.estimate_distance(box_width)
            distances.append(distance)
        return distances

    def draw_distance(self, frame, bounding_boxes, distances):
        """
        Draws the estimated distances on the frame for each detected object.

        Parameters:
        - frame: The input image frame.
        - bounding_boxes: A list of bounding box tuples (x, y, w, h).
        - distances: A list of distances corresponding to each bounding box.

        Returns:
        - frame: The frame with annotated distances.
        """
        for (x, y, w, h), distance in zip(bounding_boxes, distances):
            if distance is not None:
                cv2.putText(frame, f"{distance:.2f} m", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return frame

    def verbose_mode(self, enable):
        """
        Enables or disables verbose output.

        Parameters:
        - enable: True to enable verbose mode, False to disable it.
        """
        self.verbose = enable
