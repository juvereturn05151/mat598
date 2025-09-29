import numpy as np

class CustomLinearRegression:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.beta = 0

    def train_linear_regression(self):
        X_b = np.c_[np.ones((self.x_train.shape[0], 1)), self.x_train]
        self.beta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ self.y_train

    def predict(self):
        X_b = np.c_[np.ones((self.x_test.shape[0], 1)), self.x_test]
        return X_b @ self.beta

    def getRMSE(self,y_pred):
        return np.sqrt(np.mean((self.y_test - y_pred) ** 2))