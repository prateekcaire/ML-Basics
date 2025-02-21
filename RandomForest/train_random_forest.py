import numpy as np
import sklearn.datasets as datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from RandomForest import RandomForest

breast_cancer_data = datasets.load_breast_cancer()
X = breast_cancer_data.data
y = breast_cancer_data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
random_forest = RandomForest(100, 10)
random_forest.fit(X_train, y_train)
predictions = random_forest.predict(X_test)

plt.style.use("seaborn-darkgrid")
plt.xlabel(breast_cancer_data.feature_names[1])
plt.ylabel('has cancer')
cmap = plt.get_cmap("summer")
plt.scatter(X_train[:, 2], y_train, color=cmap(0.9), label="train")
correct_indices = np.where(predictions == y_test)[0]
plt.scatter(X_test[correct_indices, 2], y_test[correct_indices], color="green", label="correct prediction")
incorrect_indices = np.where(predictions != y_test)[0]
plt.scatter(X_test[incorrect_indices, 2], y_test[incorrect_indices], color="red", label="incorrect prediction")
plt.legend()
plt.savefig("../images/rand_forest.png")


def accuracy(predictions, y_test):
    return np.sum(predictions == y_test)/len(y_test)


acc = accuracy(predictions, y_test)
print(acc)


