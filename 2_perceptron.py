""" perceptron
    ----------
    Implementation of a single layer perceptron.

"""


# IMPORT LIBRARIES AND/OR MODULES


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as clr


# DESIGN THE PERCEPTRON


class Perceptron(object):

    """ Perceptron classifier

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
    n_miscl_ : list
        Number of n_misclassifications (updates) in each epoch.

    """

    def __init__(self, eta=0.01, n_epoch=50, random_state=1):
        self.eta = eta
        self.n_epoch = n_epoch
        self.random_state = random_state

    def fit(self, X, y):

        """ Fit training data

        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns:
        --------
        self : object

        """

        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.n_miscl_ = []

        for i in range(self.n_epoch):
            n_miscl = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[0] += update
                self.w_[1:] += update * xi
                n_miscl = np.where(update != 0, n_miscl + 1, n_miscl)
            self.n_miscl_.append(n_miscl)

        return self

    def net_input(self, X):

        """ Return net input """

        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):

        """ Return class label after unit step """

        return np.where(self.net_input(X) >= 0.0, 1, -1)


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


# Initialize a perceptron object

ppn = Perceptron(eta=0.1, n_epoch=10)


# Learn from data via the fit method (predict method is called in fit method for weight update)

ppn.fit(X, y)


# Plot the number of n_misclassifications per epoch

plt.figure()
plt.plot(range(1, len(ppn.n_miscl_)+1), ppn.n_miscl_, marker="o")
plt.xlabel("Epochs")
plt.ylabel("Number of n_misclassifications")


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

    plt.xlabel("Sepal length [cm]")
    plt.ylabel("Petal length [cm]")
    plt.legend(loc="upper left")


# APPLY THE PERCEPTRON


plot_decision_regions(X, y, classifier=ppn)


# Show plots

plt.show()

