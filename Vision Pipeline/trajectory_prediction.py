import numpy as np
from sklearn.linear_model import LinearRegression

class TrajectoryPredictor:
    def __init__(self):
        self.positions = []  # List of (x, y) tuples
    
    def update_positions(self, position):
        self.positions.append(position)
        if len(self.positions) > 10:  # Limit -mto the last 10 positions
            self.positions.pop(0)
    
    def predict_landing(self, floor_y):
        if len(self.positions) < 3:  # Not enough data to predict
            return None
        
        recent_positions = np.array(self.positions)
        x_vals = recent_positions[:, 0].reshape(-1, 1)
        y_vals = recent_positions[:, 1]
        
        model = LinearRegression()
        model.fit(x_vals, y_vals)
        predicted_x = model.predict([[floor_y]])
        
        return int(predicted_x[0])
