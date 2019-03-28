# IMPORT LIBRARIES AND/OR MODULES


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# DESIGN THE ADALINE


class AdalineGD(object):

    """ ADAptive LInear NEuron classifier

    Parameters:
    -----------
    eta : float
        Learning rate.
    n_epoch : int
        Passes over the training set.
    random_state : int
        Random number generator seed for random weight initialization.

    Attributes:
    -----------
    w_ : 1d-array
        Weights after fitting.
    cost_ : list
        Sum of squares cost function value in each epoch

    """

    def __init__(self, eta=0.01, n_epoch=50, random_state=1):
        self.eta = eta
        self.n_epoch = n_epoch
        self.random_state = random_state

    def fit(self, X, y):

        """ Fit training data

         Parameters:
        -----------
        x : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and n_features is the number of features.
        Y : array-like shape = [n_samples]
            Target values.

        Returns:
        --------
        self : object

        """

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_epoch):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y-output)
            self.w_[0] += self.eta*errors.sum()
            self.w_[1:] += self.eta*X.T.dot(errors)
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)

        return self

    def net_input(self, X):

        """ Return net input """

        return np.dot(X, self.w[1:]) + self.w_[0]

    def activation(self, X):

         """ Return linear activation """

         return X

    def predict(self, X):

        """ Return class label after unit step """

        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


# TRAIN THE PERCEPTRON


# Import the dataset

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
print(data.head())


# Extract the class labels

y = data.iloc[0:100, 4].to_numpy()
y = np.where(y == "Iris-setosa", -1, 1)


# Extract the features

X = data.iloc[0:100, [0, 2]].to_numpy()


# Plot the cost against the number of epochs for two different learning rates

fig, ax = plt.subplots(nrows=1, ncols=2)

ada1 = AdalineGD(n_epoch=10, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) +1), np.log10(ada1.cost_), marker="o")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("log(Sum-squared-errors)")
ax[0].set_title("Adaline - Learning rate 0.01")

ada2 = AdalineGD(n_epoch=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) )+1, np.log10(ada2.cost_), marker="o")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("log(Sum-squared-errors)")
ax[1].set_title("Adaline - Learning rate 0.0001")

plt.show()

