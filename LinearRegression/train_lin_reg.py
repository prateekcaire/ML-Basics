import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split

from LinearRegression import LinearRegression

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# plt.show()

linear_regression = LinearRegression(lr=0.01, n_iters=1000)
linear_regression.fit(X_train, y_train)
predictions = linear_regression.predict(X_test)

accuracy = np.mean((predictions - y_test) ** 2)
print(accuracy)

plt.style.use('seaborn-darkgrid')
plt.figure()
cmap = plt.get_cmap("summer")
plt.scatter(X_train[:, 0], y_train, color=cmap(0.7), label='train')
plt.scatter(X_test[:, 0], y_test, color=cmap(0.4), label='test')
plt.style.use('seaborn-darkgrid')
plt.title("Synthetic data")
plt.xlabel('Independent variable x')
plt.ylabel('Dependent variable y')

predictions = linear_regression.predict(X)
plt.plot(X, predictions, color='orange', label="Prediction")
plt.legend()
plt.savefig('linreg.png')
