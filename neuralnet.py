import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

class NeuralNetWrapper(object):

    def __init__(self, hidden_layer_sizes=(100,), learning_rate_init=0.001, max_iter=300):
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=42
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)

    def get_params(self):
        return self.model.get_params()


def test_neural_net(hidden_layers, lr, max_iter, X_train, y_train, X_test, y_test):
    model = NeuralNetWrapper(hidden_layer_sizes=hidden_layers, learning_rate_init=lr, max_iter=max_iter)
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)

    test_preds = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    precision = precision_score(y_test, test_preds, pos_label=1)

    cm = confusion_matrix(y_test, test_preds, labels=[1, -1])
    print(cm)

    return model.get_params(), train_acc, test_acc, precision


def test_accuracy(X_train, y_train, X_test, y_test):
    settings = [
        ((50,), 0.001, 300)#,
        #((100,), 0.01, 500),
        #((64, 32), 0.001, 300),
        #((128, 64, 32), 0.0005, 500),
    ]

    for i, (hidden_layers, lr, max_iter) in enumerate(settings):
        _, train_acc, test_acc, precision = test_neural_net(hidden_layers, lr, max_iter, X_train, y_train, X_test, y_test)
        print(f"Case {i+1}: layers={hidden_layers}, lr={lr}, max_iter={max_iter} → train acc={train_acc:.4f}, test acc={test_acc:.4f}, precision={precision:.4f}")

    print("Neural Net accuracy testing done.")


# === Load and prepare data ===
# For manually selected dataset:
train = pd.read_csv('loan_data_train.csv')
test = pd.read_csv('loan_data_test.csv')

# For forward selection dataset:
# train = pd.read_csv('fwd_train.csv')
# test = pd.read_csv('fwd_test.csv')

# For PCA dataset:
train = pd.read_csv('pca_train.csv')
test = pd.read_csv('pca_test.csv')

train = np.array(train, dtype=float)
test = np.array(test, dtype=float)

y_train = train[:, 0]
X_train = train[:, 1:]

y_test = test[:, 0]
X_test = test[:, 1:]

# Run test
test_accuracy(X_train, y_train, X_test, y_test)