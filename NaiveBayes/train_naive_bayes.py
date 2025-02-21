import numpy as np
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from NaiveBayes import NaiveBayes

X, y = datasets.make_classification(n_samples= 1000, n_features= 10, n_classes=2, random_state=123)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

naive_bayes = NaiveBayes()
naive_bayes.fit(X_train, y_train)
predictions = naive_bayes.predict(X_test)
accuracy = np.sum(predictions == y_test)/len(y_test)
print(accuracy)

