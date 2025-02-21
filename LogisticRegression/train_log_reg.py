import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression

bc = datasets.load_breast_cancer()
X = bc.data
y = bc.target

# understand the data before use
df = pd.DataFrame(bc.data, columns=bc.feature_names)
df['target'] = bc.target
print(df.columns)
print(df.head(2000))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logistic_regression = LogisticRegression(lr=0.01, n_iters=1000)
logistic_regression.fit(X_train, y_train)
predictions = logistic_regression.predict(X_test)

plt.style.use("seaborn-darkgrid")
plt.xlabel(bc.feature_names[1])
plt.ylabel('has cancer')
cmap = plt.get_cmap("summer")
plt.scatter(X_train[:, 2], y_train, color=cmap(0.9), label="train")
correct_indices = np.where(predictions == y_test)[0]
plt.scatter(X_test[correct_indices, 2], y_test[correct_indices], color="green", label="correct prediction")
incorrect_indices = np.where(predictions != y_test)[0]
plt.scatter(X_test[incorrect_indices, 2], y_test[incorrect_indices], color="red", label="incorrect prediction")
plt.legend()
plt.savefig("../images/logreg.png")

accuracy = np.sum(predictions == y_test)/len(y_test)
print(accuracy)
