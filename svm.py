import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

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

# For manually selected dataset:
train = pd.read_csv('loan_data_train.csv')
test = pd.read_csv('loan_data_test.csv')

# For forward selection dataset:
# train = pd.read_csv('fwd_train.csv')
# test = pd.read_csv('fwd_test.csv')

# For PCA dataset:
# train = pd.read_csv('pca_train.csv')
# test = pd.read_csv('pca_test.csv')

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
        precision = precision_score(y_test, predictions)
        outputstr3 = f"Precision: {precision}"
        print(outputstr3)
        file.write(outputstr1+"\n"+outputstr2+"\n"+outputstr3+"\n\n")


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
#runSVM(0.0001, 0.001, 100)      
# print("Changing lambda param")
#runSVM(0.001, 0.02, 100)      
#runSVM(0.001, 0.005, 100)     
#runSVM(0.001, 0.01, 100)      
# print("Changing n_iter")
#runSVM(0.001, 0.01, 50)       
#runSVM(0.001, 0.01, 100)      
#runSVM(0.001, 0.01, 200)     
def baggingSVM(n_models, p_data, learning_rate, lambda_param, n_iters):
    finalpredictions = np.zeros((n_models, len(y_test)))
    for i in range(n_models):
        # Compute number of samples to select
        num_samples = int(p_data * X_train.shape[0])
        # Randomly select indices with replacement
        indices = np.random.randint(0, X_train.shape[0], size=num_samples)     
        # Create subset using selected indices
        subsetX = X_train[indices]
        subsetY = y_train[indices]
        svm = SVM(learning_rate, lambda_param, n_iters)
        svm.fit(subsetX, subsetY)
        predictions = svm.predict(X_test)
        outputstr = "Test accuracy of model based on learning rate, "+str(learning_rate)+", lambda param of "+str(lambda_param)+", and n_iters of "+str(n_iters)+": "+str(testAccuracy(y_test, predictions))
        print(outputstr)
        finalpredictions[i]= predictions
    # Compute the mean of votes along axis=0 (across models for each sample)
    mean_votes = np.mean(finalpredictions, axis=0)

    finalprediction = np.where(mean_votes < 0, -1, 1)
    print("Bagged test accuracy: "+ str(testAccuracy(y_test, finalprediction)))
    precision = precision_score(y_test, predictions)
    outputstr3 = f"Precision: {precision}"
    print(outputstr3)

def boostingSVM(n_models, learning_rate, lambda_param, n_iters):
    n_samples = X_train.shape[0]
    # Initialize uniform sample weights
    sample_weights = np.ones(n_samples) / n_samples
    alphas = []   # Model weights in the final classifier
    models = []   # List to store trained SVM models
    
    for t in range(n_models):
        # Sample training data with replacement according to sample_weights
        indices = np.random.choice(np.arange(n_samples), size=n_samples, replace=True, p=sample_weights)
        subsetX = X_train[indices]
        subsetY = y_train[indices]
        
        # Train SVM on the weighted sample
        svm = SVM(learning_rate, lambda_param, n_iters)
        svm.fit(subsetX, subsetY)
        
        # Get predictions on the entire training set
        predictions_train = svm.predict(X_train)
        
        # Calculate the weighted error rate
        incorrect = (predictions_train != y_train)
        error = np.sum(sample_weights * incorrect) / np.sum(sample_weights)
        
        # Avoid division by zero or extreme values
        error = np.clip(error, 1e-10, 0.5)
        
        # Compute the model's weight (alpha)
        alpha = 0.5 * np.log((1 - error) / error)
        alphas.append(alpha)
        models.append(svm)
        
        # Update sample weights: increase weight for misclassified examples
        sample_weights *= np.exp(-alpha * y_train * predictions_train)
        sample_weights /= np.sum(sample_weights)  # Normalize
        
        # For monitoring, check test accuracy at each iteration
        test_pred = svm.predict(X_test)
        print(f"Iteration {t+1}, Weighted error: {error:.4f}, Alpha: {alpha:.4f}, Test accuracy: {testAccuracy(y_test, test_pred)}")
        
    
    # Final prediction: sum of weighted predictions from each model
    agg_preds = np.zeros(len(y_test))
    for alpha, model in zip(alphas, models):
        agg_preds += alpha * model.predict(X_test)
    
    final_predictions = np.where(agg_preds >= 0, 1, -1)
    print("Boosted test accuracy:")
    print(str(testAccuracy(y_test, final_predictions)))
    
    # Optional: Print precision if needed
    precision = precision_score(y_test, final_predictions)
    print(f"Precision: {precision:.4f}")

#boostingSVM(8,0.0001, 0.001, 100)

