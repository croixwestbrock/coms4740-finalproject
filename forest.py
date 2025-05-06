import numpy as np
import pandas as pd
import sys
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

class RandomForestWrapper(object):

    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def fit(self, X, y):
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=42)
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)

    def get_params(self):
        return self.model.get_params()


def test_random_forest(n_estimators, max_depth, X_train, y_train, X_test, y_test):
    model = RandomForestWrapper(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)

    test_preds = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    precision = precision_score(y_test, test_preds, pos_label=1)

    cm = confusion_matrix(y_test, test_preds, labels=[1, -1])
    print(cm)

    return model.get_params(), train_acc, test_acc, precision


def test_accuracy(X_train, y_train, X_test, y_test):
    #estimator_settings = [(10, None), (50, None), (100, None),(100, 3), (100, 5), (100, 10), (200, None), (200, 5), (200, 10)]
    estimator_settings = [(100, 10)]

    for i, (n_estimators, max_depth) in enumerate(estimator_settings):
        _, train_acc, test_acc, precision = test_random_forest(n_estimators, max_depth, X_train, y_train, X_test, y_test)
        print(f"Case {i+1}: n_estimators={n_estimators}, max_depth={max_depth}, train accuracy={train_acc:.4f}, test accuracy={test_acc:.4f}, precision={precision:.4f}")

    print("Random Forest accuracy testing done.")


# For manually selected dataset:
train = pd.read_csv('loan_data_train.csv')
test = pd.read_csv('loan_data_test.csv')

# For forward selection dataset:
# train = pd.read_csv('fwd_train.csv')
# test = pd.read_csv('fwd_test.csv')

# For PCA dataset:
# train = pd.read_csv('pca_train.csv')
# test = pd.read_csv('pca_test.csv')

train = np.array(train, dtype=float)
test = np.array(test, dtype=float)

y_train = train[:, 0]
X_train = train[:, 1:]

y_test = test[:, 0]
X_test = test[:, 1:]

test_accuracy(X_train, y_train, X_test, y_test)