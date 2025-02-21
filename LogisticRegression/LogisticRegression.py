import numpy as np


class LogisticRegression:
  def __init__(self, lr, n_iters):
    self.lr = lr
    self.n_iters = n_iters
    self.weights = None
    self.bias = None

  def fit(self, X, y):
    n_samples, n_features = X.shape
    self.weights = np.zeros(n_features)
    self.bias = 0

    for _ in range(self.n_iters):
      linear_y_hat = np.dot(X, self.weights) + self.bias
      y_hat = 1 / (1 + np.exp(-linear_y_hat))
      dw = (1 / n_samples) * np.dot(X.T, (y_hat - y))
      db = np.mean(y_hat - y)

      self.weights = self.weights - self.lr * dw
      self.bias = self.bias - self.lr * db

  def predict(self, X):
    linear_y_hat = np.dot(X, self.weights) + self.bias
    y_hat = 1 / (1 + np.exp(-linear_y_hat))
    class_pred = [1 if y >= 0.5 else 0 for y in y_hat]
    return class_pred
