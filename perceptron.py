import numpy as np
import pandas as pd
import sys
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix

class Perceptron(object):
    
    def __init__(self, max_iter):
        self.max_iter = max_iter

    def fit(self, X, y):
        n_samples, n_features = X.shape

        w = np.zeros(n_features)
        
        for maxiters in range(self.max_iter):
            for t in range(n_samples): 
                w_nextiter = w
                if np.sign(np.dot(w, X[t])) != y[t]: 
                    weight_multiplier = 4 if y[t] == 1 else 1
                    w_nextiter = w + weight_multiplier * y[t] * X[t]
                    #w_nextiter = w + (y[t] * X[t])
                w = w_nextiter
        
        self.W = w
        return self
    

    def get_params(self):
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W
    

    def predict(self, X):
        preds = np.sign(np.dot(X, self.W))
        return preds
        

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


def test_perceptron(max_iter, X_train, y_train, X_test, y_test):

    # train perceptron
    model = Perceptron(max_iter)
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    W = model.get_params()

    # test perceptron model
    #test_acc = model.score(X_test, y_test)
    test_preds = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    precision = precision_score(y_test, test_preds)

    cm = confusion_matrix(y_test, test_preds, labels=[1, -1])
    print(cm)

    return W, train_acc, test_acc, precision


def test_accuracy(X_train, y_train, X_test, y_test):
	max_iter = [10, 30, 50, 100, 200, 1000]
	for i, m_iter in enumerate(max_iter):
		_, train_acc, test_acc, precision = test_perceptron(m_iter, X_train, y_train, X_test, y_test)

		print("Case %d: max iteration:%d  train accuracy:%f  test accuracy: %f  precision: %f." %(i+1, m_iter, train_acc, test_acc, precision))
        

	print("Accuracy testing done.")


# For manually selected dataset:
train = pd.read_csv('loan_data_train.csv')
test = pd.read_csv('loan_data_test.csv')

# For forward selection dataset:
#train = pd.read_csv('fwd_train.csv')
#test = pd.read_csv('fwd_test.csv')

# For PCA dataset:
#train = pd.read_csv('pca_train.csv')
#test = pd.read_csv('pca_test.csv')

# Convert to NumPy arrays
train = np.array(train, dtype=float) # Convert all values to float
test = np.array(test, dtype=float)

y_train = train[:, 0]  # First column as target
X_train = train[:, 1:]  # Rest as features

y_test = test[:, 0] 
X_test = test[:, 1:] 

test_accuracy(X_train, y_train, X_test, y_test)

# # Train Perceptron
# perceptron = Perceptron(100)
# perceptron.fit(X_train, y_train)

# # Make predictions and test accuracy
# predictions = perceptron.predict(X_test)
# print("Predictions:", predictions)
# testAccuracy(y_test, predictions)