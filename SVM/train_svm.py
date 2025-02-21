from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
from SVM import SVM
import matplotlib.pyplot as plt

X, y = datasets.make_blobs(
    n_samples=300, n_features=2, centers=2, cluster_std=1.05, random_state=123
)
y = np.where(y == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

svm = SVM()
svm.fit(X_train, y_train)
predictions = svm.predict(X_test)


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


print("SVM classification accuracy", accuracy(y_test, predictions))

# visualize SVM
plt.figure()
plt.style.use('seaborn-darkgrid')
for _, label in enumerate([-1, 1]):
    plt.scatter(X_train[y_train == label, 0], X_train[y_train == label, 1], edgecolors='black', cmap='summer',
                label="train " + str(label))
for _, label in enumerate(np.unique(y)):
    plt.scatter(X_test[y_test == label, 0], X_test[y_test == label, 1], edgecolors='black', cmap='summer',
                label="test " + str(label))

x0_1 = np.amin(X[:, 0])
x0_2 = np.amax(X[:, 1])


def hyperplane_value(x, w, b, offset):
    return (-w[0] * x + b + offset) / w[1]


x1_1 = hyperplane_value(x0_1, svm.w, svm.b, 0)
x1_2 = hyperplane_value(x0_2, svm.w, svm.b, 0)
plt.plot([x0_1, x0_2], [x1_1, x1_2], color='orange', label="SVM Line")

x1_1_p = hyperplane_value(x0_1, svm.w, svm.b, 1)
x1_2_p = hyperplane_value(x0_2, svm.w, svm.b, 1)
plt.plot([x0_1, x0_2], [x1_1_p, x1_2_p], color='green', label="SVM Line Positive")

x1_1_n = hyperplane_value(x0_1, svm.w, svm.b, -1)
x1_2_n = hyperplane_value(x0_2, svm.w, svm.b, -1)
plt.plot([x0_1, x0_2], [x1_1_n, x1_2_n], color='yellow', label="SVM Line Negative")

plt.legend()
plt.savefig("../images/svm.png")
