import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from DecisionTree import DecisionTree
from DecisionTree import Node

bc_data = datasets.load_breast_cancer()
features = bc_data.feature_names
decision_tree = DecisionTree(depth=30)
X_train, X_test, y_train, y_test = train_test_split(bc_data.data, bc_data.target, test_size=0.2, random_state=42)
root = decision_tree.fit(X_train, y_train)
previous = None

count = 0
def plot_inorder(root:Node, bc, X, y):
    global count
    if root:
        plot_inorder(root.left, bc, X, y)
        #plot the chart
        split, min_y, max_y, min_x, max_x = root.split, root.min_y, root.max_y, root.min_x, root.max_x
        x_feature_index, y_feature_index = root.feature_index, root.prev_feature_index
        for i, target_name in enumerate(bc.target_names):
            plt.scatter(X[y == i, x_feature_index], X[y == i, y_feature_index], edgecolors='k', cmap="summer", label=target_name)
        plt.plot([split, split], [min_y, max_y], color='orange', label=split)
        plt.ylim(min_y, max_y)
        plt.xlim(min_x, max_x)
        if x_feature_index is not None and y_feature_index is not None:
            plt.xlabel(bc.feature_names[x_feature_index])
            plt.ylabel(bc.feature_names[y_feature_index])
        plt.legend()
        plt.savefig('../images/DecisionTree/chart'+ str(count) + '.png')
        plt.clf()
        count = count + 1
        plot_inorder(root.right, bc, X, y)


plot_inorder(root,bc_data, bc_data.data, bc_data.target)
predictions = decision_tree.predict(X_test)
acc = np.sum(predictions == y_test) / len(predictions)
print(acc)
