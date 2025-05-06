import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

# For manually selected dataset:
train = pd.read_csv('loan_data_train.csv')
test = pd.read_csv('loan_data_test.csv')

# For forward selection dataset:
# train = pd.read_csv('fwd_train.csv')
# test = pd.read_csv('fwd_test.csv')

# For PCA dataset:
# train = pd.read_csv('pca_train.csv')
# test = pd.read_csv('pca_test.csv')

y_train = train.iloc[:, 0] 
X_train = train.iloc[:, 1:]  

y_test = test.iloc[:, 0] 
X_test = test.iloc[:, 1:] 

model = LogisticRegression()
model.fit(X_train, y_train)

train_preds = model.predict(X_train)
train_acc = accuracy_score(y_train, train_preds)

test_preds = model.predict(X_test)
test_acc = accuracy_score(y_test, test_preds)
precision = precision_score(y_test, test_preds)

print(f"Training Accuracy: {train_acc}")
print(f"Testing Accuracy: {test_acc}")
print(f"Precision: {precision}")


# from sklearn.model_selection import GridSearchCV
# param_grid = {
#     'C': [0.01, 0.1, 1, 10],
#     'max_iter': [100, 500, 1000]
# }

# grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
# grid.fit(X_train, y_train)

# print(grid.best_params_)