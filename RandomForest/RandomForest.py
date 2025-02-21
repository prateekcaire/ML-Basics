from collections import Counter
import numpy as np
from DecisionTree import DecisionTree

class RandomForest:
    def __init__(self, n_trees=100, max_depth=10):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_trees):
            tree = DecisionTree(self.max_depth)
            X_sample, y_sample = self.generate_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        preds = [tree.predict(X) for tree in self.trees]
        preds = np.swapaxes(preds, 0, 1)
        return [self.most_common_label(pred) for pred in preds]

    def generate_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, 80, replace=False)
        return X[idxs], y[idxs]

    def most_common_label(self, labels):
        counter = Counter(labels)
        return counter.most_common(1)[0][0]