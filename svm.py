import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

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
    return accuracy


# Example usage:

train = pd.read_csv('loan_data_train.csv')
test = pd.read_csv('loan_data_test.csv')

# Convert to NumPy arrays
train = np.array(train, dtype=float) # Convert all values to float
test = np.array(test, dtype=float)

y_train = train[:, 0]  # First column as target
X_train = train[:, 1:]  # Rest as features

y_test = test[:, 0] 
X_test = test[:, 1:] 


# Train SVM with revised hyperparameters
def runSVMandLog(learning_rate, lambda_param, n_iters):
    with open("SVMoutputs.txt", "a") as file:
        svm = SVM(learning_rate, lambda_param, n_iters)
        svm.fit(X_train, y_train)
        trainpreds = svm.predict(X_train)
        trainacc = testAccuracy(y_train, trainpreds)
        outputstr1 = "Train accuracy of model based on learning rate, "+str(learning_rate)+", lambda param of "+str(lambda_param)+", and n_iters of "+str(n_iters)+": "+str(trainacc)
        print(outputstr1)
        predictions = svm.predict(X_test)
        outputstr2 = "Test accuracy of model based on learning rate, "+str(learning_rate)+", lambda param of "+str(lambda_param)+", and n_iters of "+str(n_iters)+": "+str(testAccuracy(y_test, predictions))
        print(outputstr2)
        file.write(outputstr1+"\n"+outputstr2+"\n\n")


def runSVM(learning_rate, lambda_param, n_iters):
    svm = SVM(learning_rate, lambda_param, n_iters)
    svm.fit(X_train, y_train)
    predictions = svm.predict(X_test)
    outputstr = "Test accuracy of model based on learning rate, "+str(learning_rate)+", lambda param of "+str(lambda_param)+", and n_iters of "+str(n_iters)+": "+str(testAccuracy(y_test, predictions))
    print(outputstr)

#runSVMandLog(0.0005, 0.005, 100)    #0.8634888888888889
#runSVMandLog(0.0003, 0.003, 100)    #0.8828666666666667
runSVMandLog(0.0001, 0.001, 100)
# print("Changing learning rate")
#runSVM(0.002, 0.01, 100)      #0.7720666666666667
#runSVM(0.0005, 0.01, 100)     
#runSVM(0.001, 0.01, 100)      
# print("Changing lambda param")
#runSVM(0.001, 0.02, 100)      
#runSVM(0.001, 0.005, 100)     
#runSVM(0.001, 0.01, 100)      
# print("Changing n_iter")
#runSVM(0.001, 0.01, 50)       
#runSVM(0.001, 0.01, 100)      
#runSVM(0.001, 0.01, 200)      

