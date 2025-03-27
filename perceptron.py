import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

class Perceptron:
    def __init__(self, learning_rate=0.25, epochs=10000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                prediction = self.predict_single(x_i)
                update = self.learning_rate * (y[idx] - prediction)
                self.weights += update * x_i
                self.bias += update
    
    def predict_single(self, x):
        return 1 if np.dot(x, self.weights) + self.bias >= 0 else 0
    
    def predict(self, X):
        return np.array([self.predict_single(x) for x in X])


def testAccuracy(y, predictedY):
    accuracy = accuracy_score(y, predictedY)
    print("Accuracy of model:", accuracy)

# Read and parse the file correctly
file_path = "loan_data_preprocessed.csv"

with open(file_path, "r") as file:
    lines = file.readlines()

# Extract header and data
header = lines[0].strip().split(",")  # First line is column names
data = [line.strip().split(",") for line in lines[1:]]  # Skip the header

# Convert to NumPy arrays
data = np.array(data, dtype=float)  # Convert all values to float
y = data[:, 0]  # First column as target
X = data[:, 1:]  # Rest as features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train Perceptron
perceptron = Perceptron(learning_rate=0.1, epochs=1)
perceptron.fit(X, y)

# Make predictions and test accuracy
predictions = perceptron.predict(X)
print("Predictions:", predictions)
testAccuracy(y, predictions)