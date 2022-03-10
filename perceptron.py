""" PERCEPTRON
    ----------
    Implementation of a single layer perceptron for binary classification.
"""


# -------------------------------------------------------------------------------
# 0. IMPORT LIBRARIES
# -------------------------------------------------------------------------------


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as clr


# -------------------------------------------------------------------------------
# 1. PREPARE THE DATA
# -------------------------------------------------------------------------------


# Import the dataset

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
print(data.head())


# Extract the class labels

y = data.iloc[:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)


# Extract the features

X = data.iloc[:100, [0, 2]].values


# Plot the features in a scatter plot

plt.figure()
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='+', label='Setosa')
plt.scatter(X[50:, 0], X[50:, 1], color='blue', marker='+', label='Versicolor')
plt.title('Scatter plot of the features')
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.savefig('images/01_perceptron/Scatter_plot_of_the_features.png')


# -------------------------------------------------------------------------------
# 2. DESIGN THE MODEL
# -------------------------------------------------------------------------------


# Design the perceptron

class Perceptron:

    """ Perceptron classifier

        Parameters:
        ----------
        eta : float
            Learning rate (between 0.0 and 1.0).
        n_epochs : int
            Number of epochs.

        Attributes:
        ----------
        w : array, shape = [n_features, ]
            Weights after fitting.
        b : scalar
            Bias unit after fitting.
        n_misclass : list
            Number of misclassifications (hence weight updates) in each epoch.
    """

    def __init__(self, eta=0.01, n_epochs=100):

        self.eta = eta
        self.n_epochs = n_epochs

    def fit(self, X, y):

        """ Fit training set

            Parameters:
            ----------
            X : array, shape = [n_samples, n_features]
                Training vectors.
            y : array, shape = [n_samples, ]
                Target values.

            Returns:
            -------
            self : object
        """

        rgen = np.random.RandomState(seed=1)
        self.w = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b = 0.0
        self.n_misclass = []

        for epoch in range(self.n_epochs):

            misclass = 0

            for Xi, yi in zip(X, y):

                update = yi - self.step_activ(Xi)
                self.w += self.eta * update * Xi
                self.b += self.eta * update
                misclass += int(update != 0)

            self.n_misclass.append(misclass)

        return self

    def step_activ(self, X):

        """ Calculate the net input and return the class label prediction after the unit step function
            (Used in the fit method and in plot_decision_regions function)

            Parameters:
            ----------
            X : array, shape = [n_features, ] in fit method
                array, shape = [X0X1_combs.shape[0], n_features] in plot_decision_regions function

            Returns:
            -------
            step_activ : int in fit method
                         array, shape = [X0X1_combs.shape[0], ] in plot_decision_regions function
        """

        net_input = np.dot(X, self.w) + self.b

        return np.where(net_input >= 0.0, 1, 0)


# -------------------------------------------------------------------------------
# 3. TRAIN THE MODEL
# -------------------------------------------------------------------------------


# Initialize a perceptron object

ppn = Perceptron(eta=0.1, n_epochs=10)


# Learn from the data via the fit method

ppn.fit(X, y)


# Plot the number of misclassifications per epoch

plt.figure()
plt.plot(range(1, len(ppn.n_misclass) + 1), ppn.n_misclass, marker='o')
plt.title('Number of misclassifications per epoch')
plt.xlabel('Epoch')
plt.ylabel('Number of misclassifications')
plt.savefig('images/01_perceptron/Number_of_misclassifications_per_epoch.png')


# -------------------------------------------------------------------------------
# 4. EVALUATE THE MODEL
# -------------------------------------------------------------------------------


# Function to plot the decision boundary

def plot_decision_regions(X, y, classifier, resolution=0.02):

    """ Create a colormap from the list of colors.

        Generate a matrix with two columns, where rows are all possible combinations of all numbers from min-1 to max+1 of the two series of
        features. A matrix with two columns is needed because the perceptron was trained on a matrix with such shape.

        Use the step_activ method of the ppn to predict the class corresponding to all the possible combinations of features generated in the
        above matrix. The step_activ method will use the weights learnt during the training phase: since the number of misclassifications con-
        verged during the training phase, we expect the perceptron to find a decision boundary that correctly classifies all the samples in
        the training set.

        Reshape the vector of predictions as the X0_grid.

        Draw filled contours, where all possible combinations of features are associated to a Z, which is 0 or 1.

        To verify that the perceptron correctly classified all the samples in the training set, plot the original features in the scatter
        plot and verify that they fall inside the correct region.
    """

    colors = ('red', 'blue', 'green')
    cmap = clr.ListedColormap(colors[:len(np.unique(y))])

    X0_min, X0_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    X1_min, X1_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    X0_grid, X1_grid = np.meshgrid(np.arange(X0_min, X0_max, resolution), np.arange(X1_min, X1_max, resolution))
    X0X1_combs = np.array([X0_grid.ravel(), X1_grid.ravel()]).T

    Z = classifier.step_activ(X0X1_combs)

    Z = Z.reshape(X0_grid.shape)

    plt.figure()
    plt.contourf(X0_grid, X1_grid, Z, alpha=0.3, cmap=cmap)
    plt.xlim(X0_min, X0_max)
    plt.ylim(X1_min, X1_max)

    for pos, cl in enumerate(np.unique(y)):

        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, color=colors[pos], marker='+', label=cl)


# Plot the decision region and the data

plot_decision_regions(X, y, classifier=ppn)
plt.title('Decision boundary and training sample')
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.savefig('images/01_perceptron/Decision_boundary_and_training_sample.png')


# -------------------------------------------------------------------------------
# 5. GENERAL
# -------------------------------------------------------------------------------


# Show plots

plt.show()
