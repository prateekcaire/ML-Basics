import numpy as np


class NaiveBayes:
    def __init__(self, priors=None, mean=None, labels=None, var=None):
        self.priors = priors
        self.mean = mean
        self.labels = labels
        self.var = var

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.labels = np.unique(y)
        n_labels = len(self.labels)
        self.mean = np.zeros((n_labels, n_features), dtype=np.float64)
        self.var = np.zeros((n_labels, n_features), dtype=np.float64)
        self.priors = np.zeros(n_labels, dtype=np.float64)
        for label_index in range(n_labels):
            X_label = X[y == label_index]
            self.mean[label_index, :] = np.mean(X_label, axis=0)
            self.var[label_index, :] = np.var(X_label, axis=0)
            self.priors[label_index] = X_label.shape[0] / float(n_samples)

    def predict(self, X):
        n_labels = len(self.labels)
        predictions = [self.prediction(n_labels, x) for x in X]
        return np.array(predictions)

    def prediction(self, n_labels, x):
        posteriors = []  # posteriors for each label
        for label in range(n_labels):
            prior = np.log(self.priors[label])
            likelihoods_sum = np.sum(np.log(self.pdf(x, label)))  # MLE - finding the log likelihood sum/ likelihood
            # product and then maximising log likelihood sum or use negative log likelihood as loss function
            posterior = prior + likelihoods_sum
            posteriors.append(posterior)
        return self.labels[np.argmax(posteriors)]

    def pdf(self, x, label):  # multivariate gaussian distribution for each feature. returns vector
        var = self.var[label]
        mean = self.mean[label]
        denominator = np.sqrt(2 * np.pi * var)
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        return numerator / denominator
