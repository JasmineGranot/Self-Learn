import numpy as np


class WeightedLogisticRegressionClassifier:
    def __init__(self):
        self.theta = None
        self.history = {'loss': []}

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

    def cost_function(self, x, y, sample_weight=None):
        if sample_weight is None:
            w = np.ones(y.shape)
        else:
            w = sample_weight

        m = x.shape[0]
        h = self.probability(self.theta, x)
        return -1 / m * w.T.dot(y * np.log(h) + (1 - y) * np.log(1 - h))

    def gradient(self, x, y, sample_weight):
        if sample_weight is None:
            w = np.ones(y.shape)
        else:
            w = sample_weight

        m = x.shape[0]
        h = self.sigmoid(self.net_input(self.theta, x))
        x_with_ones = np.ones((x.shape[0], x.shape[1]+1))
        x_with_ones[:, :-1] = x
        return 1 / m * np.dot(x_with_ones.T, w * (h - y))

    def predict(self, x):
        if self.theta is None:
            self.theta = np.random.randn(self.theta_size(x))
        h = self.sigmoid(self.net_input(self.theta, x))
        return np.array([1 if x_val > 0.5 else -1 for x_val in h])

    def make_gd_step(self, x, y, sample_weight, lr=0.01):
        self.theta = self.theta - lr * self.gradient(x, y, sample_weight)

    @staticmethod
    def theta_size(X):
        return X.shape[1] + 1 

    def fit(self, X, y, sample_weight=None, lr=0.01, max_iterations=100, tolerance=10 ** -3):
        if sample_weight is None:
            sample_weight = np.ones(y.shape)
        else:
            sample_weight = sample_weight
        self.theta = np.random.randn(self.theta_size(X))
        iterations = 0
        y = np.array([1 if label > 0 else 0 for label in y])
        while iterations < max_iterations and self.cost_function(X, y, sample_weight) > tolerance:
            iterations += 1
            self.history['loss'].append(self.cost_function(X, y, sample_weight))
            self.make_gd_step(X, y, sample_weight, lr)
        return self


# if __name__ == '__main__'

    # # Testing:
    # x, y = make_toy_dataset(n=10, random_seed=10)
    # plot_adaboost(x, y)
    # stump = WeightedLogisticRegressionClassifier()
    # stump = stump.fit(x, y, lr=0.1, max_iterations=100)
    # predictions = stump.predict(x)
    # loss = stump.cost_function(x, y)
    # print(f"loss = {loss}")
    # print(f"acc =  {accuracy(predictions, y)}")
    #
    # import matplotlib.pyplot as plt
    #
    # plt.plot(stump.history['loss'], label='loss')
    #
    # # Validating:
    # from sklearn.linear_model import LogisticRegression
    #
    # reg = LogisticRegression().fit(x, y)
    # predictions = stump.predict(x)
    # print(f"acc =  {accuracy(predictions, y)}")
