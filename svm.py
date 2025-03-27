import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

def testAccuracy(y, predictedY):
    accuracy = accuracy_score(y, predictedY)
    print("Accuracy of model:", accuracy)


# Example usage:

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

# Convert labels: assuming original labels are 0 and 1, convert 0 to -1 and 1 remains 1
y = np.where(y == 0, -1, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train SVM with revised hyperparameters
svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=100)
svm.fit(X, y)

# Make predictions and test accuracy
predictions = svm.predict(X)
testAccuracy(y, predictions)