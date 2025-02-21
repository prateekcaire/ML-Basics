import numpy as np


class SVM:
    def __init__(self, lr=0.1, n_iters=1000, lambda_param=0.1):
        self.lr = lr
        self.n_iters = n_iters
        self.lambda_param = lambda_param
        self.w = None
        self.b = None

    def fit1(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        condition = y * (np.dot(X, self.w) - self.b) >= 1
        hinge_loss_w_true = lambda w: 2 * self.lambda_param * self.w
        hinge_loss_w_false = lambda w, X, y: 2 * self.lambda_param * self.w - np.dot(y, X).T
        hinge_loss_b_true = lambda w: 0
        hinge_loss_b_false = lambda y: y

        dw = (1 / n_samples) * np.where(condition, hinge_loss_w_true(self.w), hinge_loss_w_false(self.w, X, y))
        db = (1 / n_samples) * np.where(condition, hinge_loss_b_true(self.w), hinge_loss_b_false(y))

        self.w = self.w - self.lr * dw
        self.b = self.b - self.lr * db

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        margin = y * (np.dot(X, self.w) - self.b)
        condition = margin >= 1

        dw = np.zeros(n_features)
        dw = 2 * self.lambda_param * self.w - np.dot(y[~condition], X[~condition]).T

        db = np.zeros(n_samples)
        db[~condition] = y[~condition]

        # Calculate the gradients of weights and bias
        dw = (1 / n_samples) * dw
        db = (1 / n_samples) * db

        # Update weights and bias
        self.w -= self.lr * dw
        self.b -= self.lr * np.mean(db)
    def predict(self, X):
        prediction = np.dot(X, self.w) - self.b
        return np.sign(prediction)
