import random
from collections import Counter

import numpy as np


class Node:
    def __init__(self, feature_index=None, split=None, left=None, right=None, value=None, prev_feature_index=None,
                 min_y=None, max_y=None, min_x=None, max_x=None):
        self.feature_index = feature_index
        self.split = split
        self.left = left
        self.right = right
        self.value = value
        self.prev_feature_index = prev_feature_index
        self.min_y = min_y
        self.max_y = max_y
        self.min_x = min_x
        self.max_x = max_x


class DecisionTree:
    def __init__(self, depth):
        self.depth = depth

    def fit(self, X, y: np.ndarray):
        n_sample, n_features = X.shape
        prev_feature_index = random.randint(0, n_features - 1)
        self.root = self.create_tree(X, y, 0, prev_feature_index)
        return self.root

    def predict(self, X):
        return [self.traverse(self.root, x) for x in X]

    def create_tree(self, X, y: np.ndarray, level, prev_feature_index):
        n_labels = len(np.unique(y))
        n_samples, n_features = X.shape
        if level >= self.depth or n_labels == 1:
            min_y, max_y = min(X[:, prev_feature_index]), max(X[:, prev_feature_index])
            return Node(value=self._most_common_label(y), min_y=min_y, max_y=max_y)
        best = {'feature': None, 'split': None, 'entropy': np.inf, 'feature_index': None}
        for feature_index in range(n_features):
            x_column = X[:, feature_index]
            splits = np.unique(x_column)
            for split in splits:
                left_indices, right_indices = self._get_split_indices(x_column, split)
                left_entropy, right_entropy = self._entropy(y[left_indices]), self._entropy(y[right_indices])
                n, n_l, n_r = len(y), len(left_indices), len(right_indices)
                entropy = (n_l / n) * left_entropy + (n_r / n) * right_entropy
                if entropy < best['entropy']:
                    best['entropy'] = entropy
                    best['feature_index'] = feature_index
                    best['split'] = split
        best_feature_index = best['feature_index']
        best_split = best['split']
        left_indices, right_indices = self._get_split_indices(X[:, best_feature_index], best_split)
        left_tree = self.create_tree(X[left_indices, :], y[left_indices], level + 1, best_feature_index)
        right_tree = self.create_tree(X[right_indices, :], y[right_indices], level + 1, best_feature_index)
        min_y, max_y = min(X[:, prev_feature_index]), max(X[:, prev_feature_index])
        min_x, max_x = min(X[:, best_feature_index]), max(X[:, best_feature_index])
        return Node(best_feature_index, best_split, left_tree, right_tree, None, prev_feature_index, min_y, max_y,
                    min_x, max_x)

    @staticmethod
    def _most_common_label(y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    @staticmethod
    def _entropy(labels):
        p = np.mean(labels)
        if p == 0 or p == 1:
            return 0
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    def traverse(self, root: Node, x):
        if root is None:
            return None
        if root.value is not None:
            return root.value
        if x[root.feature_index] <= root.split:
            return self.traverse(root.left, x)
        else:
            return self.traverse(root.right, x)

    @staticmethod
    def _get_split_indices(x_column, curr):
        left_indices = np.argwhere(x_column <= curr).flatten()
        right_indices = np.argwhere(x_column > curr).flatten()
        return left_indices, right_indices
