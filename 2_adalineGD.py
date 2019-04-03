""" adalineGD
    ---------
    Implementation of a single layer Adaptive Linear Neuron via a gradient descent algorithm. Convergence is checked
    without and with a standardization.

"""


# IMPORT LIBRARIES AND/OR MODULES


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as clr


# DESIGN THE ADALINE


class AdalineGD(object):

    """ ADAptive LInear NEuron classifier

    Parameters:
    -----------
    eta : float
        Learning rate (between 0.0 and 1.0).
    n_epoch : int
        Passes over the training dataset.
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

        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
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

        return np.dot(X, self.w_[1:]) + self.w_[0]

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


# Plot the features in a scatter plot

plt.figure()
plt.scatter(X[:50, 0], X[:50, 1], color="red", marker="o", label="Setosa")
plt.scatter(X[50:100, 0], X[50:100, 1], color="blue", marker="x", label="Versicolor")
plt.xlabel("Sepal length [cm]")
plt.ylabel("Petal length [cm]")
plt.legend(loc="upper left")


# Initialize two ADELINE objects

ada1 = AdalineGD(n_epoch=10, eta=0.01)
ada2 = AdalineGD(n_epoch=10, eta=0.0001)


# Learn from data via the fit method

ada1.fit(X, y)
ada2.fit(X, y)


# Plot the cost per epoch for two different learning rates

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

ax[0].plot(range(1, len(ada1.cost_) +1), np.log10(ada1.cost_), marker="o")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("log(Sum-squared-errors)")
ax[0].set_title("Adaline - Learning rate 0.01")

ax[1].plot(range(1, len(ada2.cost_) +1), np.log10(ada2.cost_), marker="o")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("log(Sum-squared-errors)")
ax[1].set_title("Adaline - Learning rate 0.0001")


# Apply standardization for feature scaling

X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()


# Initialize an ADELINE object

ada = AdalineGD(n_epoch=15, eta=0.01)


# Learn from data via the fit method

ada.fit(X_std, y)


# Plot the cost per epoch

plt.figure()
plt.plot(range(1, len(ada.cost_) +1), np.log10(ada.cost_), marker="o")
plt.xlabel("Epochs")
plt.ylabel("log(Sum-squared-errors)")
plt.title("Adaline - Learning rate 0.01")


# Function to visualize the decision boundaries

def plot_decision_regions(X, y, classifier, resolution=0.02):

    """ Define markers and colors and create a colormap.

        Generate a matrix with two columns where rows are all possible combinations of all numbers from min-1 to max+1
        of the two series of features.

        Predict the class label via the predict method on this matrix.

        Reshape the vector of predictions as the xx1.

        Draw filled contours, where each combination of xx1 and xx2 coordinates is associated to a Z which is +1 or -1.
        What this does is mapping the different decision regions to different colors for each predicted class.

        Plot the features in a scatter plot

    """

    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = clr.ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    Z = Z.reshape(xx1.shape)

    plt.figure()
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=colors[idx], marker=markers[idx], label=cl,
                    edgecolor='black')

    plt.xlabel("Sepal length [standardized]")
    plt.ylabel("Petal length [standardized]")
    plt.legend(loc="upper left")


# APPLY THE ADALINE


plot_decision_regions(X_std, y, classifier=ada)


# Show plots

plt.show()
