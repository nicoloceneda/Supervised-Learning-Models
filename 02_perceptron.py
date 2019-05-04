""" perceptron
    ----------
    Implementation of a single layer perceptron.
"""


# 0. IMPORT LIBRARIES AND/OR MODULES


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as clr


# 1. DESIGN THE PERCEPTRON


class Perceptron(object):

    """ Perceptron classifier

    Parameters:
    -----------
    eta : float
        Learning rate (between 0.0 and 1.0).
    n_epoch : int
        Passes over the training dataset.
    seed : int
        Random number generator seed for random weight initialization.

    Attributes:
    -----------
    w : 1d-array
        Weights after fitting.
    n_miscl : list
        Number of n_misclassifications (updates) in each epoch.
    """

    def __init__(self, eta=0.01, n_epoch=50, seed=1):

        self.eta = eta
        self.n_epoch = n_epoch
        self.seed = seed

    def fit(self, X, y):

        """ Fit training data

        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Training matrix, where n_samples is the number of samples and n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns:
        --------
        self : object
        """

        rgen = np.random.RandomState(self.seed)
        self.w = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.n_miscl = []

        for epoch in range(self.n_epoch):

            miscl = 0

            for Xi, yi in zip(X, y):

                update = self.eta * (yi - self.predict(Xi))
                self.w[0] += update
                self.w[1:] += update * Xi
                miscl += int(update != 0)

            self.n_miscl.append(miscl)

        return self

    def net_input(self, X):

        """ Return the net input """

        net_input = np.dot(X, self.w[1:]) + self.w[0]

        return net_input

    def predict(self, X):

        """ Return class label after unit step """

        prediction = np.where(self.net_input(X) >= 0.0, 1, -1)

        return prediction


# 2. PREPARE THE DATA


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


# 3. TRAIN THE PERCEPTRON


# Initialize a perceptron object

ppn = Perceptron(eta=0.1, n_epoch=10)


# Learn from data via the fit method (the predict method is called in fit method to learn the weights)

ppn.fit(X, y)


# Plot the number of n_misclassifications per epoch

plt.figure()
plt.plot(range(1, len(ppn.n_miscl) + 1), ppn.n_miscl, marker="o")
plt.xlabel("Epochs")
plt.ylabel("Number of n_misclassifications")


# 4. VISUALIZE THE DECISION BOUNDARIES AND VERIFY THAT THE TRAINING SAMPLE IS CLASSIFIED CORRECTLY


# Function to visualize the decision boundaries

def plot_decision_regions(X, y, classifier, resolution=0.02):

    """ Create a colormap.

        Generate a matrix with two columns, where rows are all possible combinations of all numbers from min-1 to max+1 of the two series of
        features. The matrix with two columns is needed because the perceptron was trained on a matrix with such shape.

        Use the predict method of the chosen classifier (ppn) to predict the class corresponding to all the possible combinations of features
        generated in the above matrix. The predict method will use the weights learnt during the training phase: since the number of mis-
        classifications converged to zero in the training phase, we expect the perceptron to correctly classify all possible combinations of
        features.

        Reshape the vector of predictions as the x0_grid.

        Draw filled contours, where all possible combinations of features are associated to a Z, which is +1 or -1.

        To verify that the perceptron correctly classified all possible combinations of the features, plot the the original features in the
        scatter plot and verify that they fall inside the correct region.
    """

    cmap = clr.ListedColormap(['red', 'blue'])

    x0_min, x0_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x1_min, x1_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x0_grid, x1_grid = np.meshgrid(np.arange(x0_min, x0_max, resolution), np.arange(x1_min, x1_max, resolution))
    x0x1_combs = np.array([x0_grid.ravel(), x1_grid.ravel()]).T

    Z = classifier.predict(x0x1_combs)

    Z = Z.reshape(x0_grid.shape)

    plt.figure()
    plt.contourf(x0_grid, x1_grid, Z, alpha=0.3, cmap=cmap)
    plt.xlim(x0_min, x0_max)
    plt.ylim(x1_min, x1_max)

    plt.scatter(X[:50, 0], X[:50, 1], alpha=0.8, color='red', marker='o', label='+1', edgecolor='black')
    plt.scatter(X[50:100, 0], X[50:100, 1], alpha=0.8, color='blue', marker='x', label='-1', edgecolor='black')
    plt.xlabel('Sepal length [cm]')
    plt.ylabel('Petal length [cm]')
    plt.legend(loc='upper left')


# Plot the decision region and the data

plot_decision_regions(X, y, classifier=ppn)


# 5. GENERAL


# Show plots

plt.show()
