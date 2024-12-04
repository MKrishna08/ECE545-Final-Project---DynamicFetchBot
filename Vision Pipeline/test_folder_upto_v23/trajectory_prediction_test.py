import numpy as np

class TrajectoryPredictor:
    def __init__(self):
        self.positions = []  # List of (x, y) tuples

    def update_positions(self, position):
        self.positions.append(position)
        if len(self.positions) > 10:  # Limit to the last 10 positions
            self.positions.pop(0)

    def predict_landing(self, floor_y):
        if len(self.positions) < 3:  # Not enough data to predict
            return None

        recent_positions = np.array(self.positions)
        x_vals = recent_positions[:, 0]
        y_vals = recent_positions[:, 1]

        # Fit a quadratic polynomial: y = ax^2 + bx + c
        coefficients = np.polyfit(y_vals, x_vals, 2)
        a, b, c = coefficients

        # Solve for x where y = floor_y (projected ground)
        predicted_x = a * floor_y**2 + b * floor_y + c

        return int(predicted_x)

    @staticmethod
    def map_to_square(real_x, square_size=2):
        """
        Map real-world coordinates to the robot's 2m x 2m square.
        Assumes the robot is at the center of the square.
        
        :param real_x: Real-world x-coordinate (in meters)
        :param square_size: Size of the square (default: 2 meters)
        :return: Mapped coordinates (x, y) relative to the square
        """
        half_square = square_size / 2
        mapped_x = real_x + half_square
        mapped_y = half_square  # Assuming constant y (e.g., floor at square center)
        return max(0, min(square_size, mapped_x)), max(0, min(square_size, mapped_y))

    def debug_coefficients(self):
        if len(self.positions) < 3:
            print("Not enough data points for prediction.")
            return

        recent_positions = np.array(self.positions)
        x_vals = recent_positions[:, 0]
        y_vals = recent_positions[:, 1]
        coefficients = np.polyfit(y_vals, x_vals, 2)
        print("Quadratic Coefficients:", coefficients)
