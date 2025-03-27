import numpy as np
import pandas as pd
import sys
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

class Perceptron(object):
    
    def __init__(self, max_iter):
        self.max_iter = max_iter
    def fit(self, X, y):
        n_samples, n_features = X.shape

        w = np.zeros(n_features)
        
        for epoch in range(self.max_iter):
            error_count = 0  
            for i in range(n_samples):
                if y[i] * np.dot(w, X[i]) <= 0:
                    w += y[i] * X[i]
                    error_count += 1

            if error_count == 0:
                break
        
        self.W = w
        return self

    def get_params(self):
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W

    def predict(self, X):
        ### YOUR CODE HERE
        preds = np.where(np.dot(X, self.W) >= 0, 1, 0)
        return preds
        
        ### END YOUR CODE

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. y.
        """
        preds = self.predict(X)
        return np.mean(preds == y)



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
perceptron = Perceptron(100)
perceptron.fit(X, y)

# Make predictions and test accuracy
predictions = perceptron.predict(X)
print("Predictions:", predictions)
testAccuracy(y, predictions)