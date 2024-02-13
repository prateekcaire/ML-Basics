import sklearn.datasets as datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from KNN import KNN

iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
plt.figure()
plt.style.use("seaborn-darkgrid")
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.title("Iris flower dataset")
for i, target_name in enumerate(iris.target_names):
    plt.scatter(X[y == i, 2], X[y == i, 3], edgecolors='k', cmap="summer", label=target_name)
# plt.show()
plt.legend()
plt.savefig("knn.png")
clf = KNN(k=5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy}")