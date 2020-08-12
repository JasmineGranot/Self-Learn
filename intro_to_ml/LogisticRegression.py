import numpy as np


class LogisticRegressionClassifier:
    def __init__(self):
        self.theta = None

    @staticmethod
    def sigmoid(x):
        # Activation function used to map any real value between 0 and 1
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def net_input(theta, x):
        # Computes the weighted sum of inputs Similar to Linear Regression
        x_with_ones = np.ones((x.shape[0], x.shape[1]+1))
        x_with_ones[:, :-1] = x
        return np.dot(x_with_ones, theta)

    def probability(self, theta, x):
        # Calculates the probability that an instance belongs to a particular class
        return self.sigmoid(self.net_input(theta, x))

    def cost_function(self, x, y):
        m = x.shape[0]
        h = self.probability(self.theta, x)
        return -1 / m * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

    def gradient(self, x, y):
        m = x.shape[0]
        h = self.probability(self.theta, x)
        x_with_ones = np.ones((x.shape[0], x.shape[1]+1))
        x_with_ones[:, :-1] = x
        return 1 / m * np.dot(x_with_ones.T, (h - y))

    def predict(self, x):
        if self.theta is None:
            self.theta = np.random.randn(self.theta_size(x))
        h = self.probability(self.theta, x)
        return np.array([1 if x_val > 0.5 else -1 for x_val in h])

    def make_gd_step(self, x, y, lr=0.01):
        self.theta = self.theta - lr * self.gradient(x, y)

    @staticmethod
    def theta_size(X):
        return X.shape[1] + 1 

    def fit(self, X, y, lr=0.01, max_iterations=100, tolerance=10 ** -3):

        self.theta = np.random.randn(self.theta_size(X))
        iterations = 0
        y = np.array([1 if label > 0 else 0 for label in y])
        while iterations < max_iterations and self.cost_function(X, y) > tolerance:
            iterations += 1
            self.make_gd_step(X, y, lr)
        return self

