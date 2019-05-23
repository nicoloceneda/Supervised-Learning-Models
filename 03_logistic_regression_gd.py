""" logistic_regression_gd
    ----------------------
    Implementation of a single layer logistic regression via gradient descent algorithm.
"""


# ------------------------------------------------------------------------------------------------------------------------------------------
# 0. IMPORT LIBRARIES AND/OR MODULES
# ------------------------------------------------------------------------------------------------------------------------------------------


import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.colors as clr


# ------------------------------------------------------------------------------------------------------------------------------------------
# 1. DESIGN THE ADALINE
# ------------------------------------------------------------------------------------------------------------------------------------------


class LogisticRegressionGD(object):

    """ Logistic Regression Classifier using gradient descent

    Parameters:
    -----------
    eta : float
        Learning rate (between 0.0 and 1.0).
    n_iter : int
        Passes over the training dataset.
    random_state : int
        Random number generator seed for random weight initialization.

    Attributes:
    -----------
    w : 1d-array
        Weights after fitting.
    cost_fun : list
        Logistic cost function value in each epoch.
    """

    def __init__(self, eta=0.05, n_iter=100, random_state=1):

        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):

        """ Fit training data

        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Training matrix, where n_samples is the number of samples and n_features is the number of features.
        y : array-like, shape = [n_samples, ]
            Target values.

        Returns:
        --------
        self : object
        """

        rgen = np.random.RandomState(self.random_state)
        self.w = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_fun = []

        for iteration in range(self.n_iter):

            update = y - (self.activation(self.net_input(X)))
            self.w[0] += self.eta * np.sum(update)
            self.w[1:] += self.eta * np.dot(X.T, update)
            cost = -y.dot(np.log(self.activation(self.net_input(X)))) - ((1 - y).dot(np.log(1 - self.activation(self.net_input(X)))))
            self.cost_fun.append(cost)

        return self

    def net_input(self, X):

        """ Return the net input """

        net_input = np.dot(X, self.w[1:]) + self.w[0]

        return net_input

    def activation(self, z):

        """ Return the logistic sigmoid actiavtion """

        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):

        """ Return the class label after unit step function """

        prediction = np.where(self.net_input(X) >= 0.0, 1, 0)

        return prediction


# ------------------------------------------------------------------------------------------------------------------------------------------
# 2. PREPARE THE DATA
# ------------------------------------------------------------------------------------------------------------------------------------------


# Import the dataset

iris = load_iris()
print(iris)


# Extract the class labels

y = iris.target[:100]


# Extract the features

X = iris.data[:100, [2, 3]]


# Plot the features in a scatter plot

plt.figure()
plt.scatter(X[:50, 0], X[:50, 1], color="red", edgecolor='black', marker="+", label="Setosa")
plt.scatter(X[50:100, 0], X[50:100, 1], color="blue", edgecolor='black', marker="+", label="Versicolor")
plt.title("Scatter plot of the features")
plt.xlabel("Petal length [cm]")
plt.ylabel("Petal width [cm]")
plt.legend(loc="upper left")
plt.savefig('images/03_logistic_regression_gd/Scatter_plot_of_the_features.png')


# Separate the data into a train and a test subset with the same proportions of class labels as the input dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)


# ------------------------------------------------------------------------------------------------------------------------------------------
# 3. TRAIN THE ADALINE
# ------------------------------------------------------------------------------------------------------------------------------------------


# Initialize the adaline object

lrgd = LogisticRegressionGD(eta=0.05, n_iter=30)


# Learn from the data via the fit method (the activation method, rather than predict method, is called in the fit method to learn the weights)

lrgd.fit(X_train, y_train)


# Plot the cost function per iter

plt.figure()
plt.plot(range(1, len(lrgd.cost_fun) + 1), lrgd.cost_fun, marker="o")
plt.title("LogisticRegressionGD")
plt.xlabel("n_iter")
plt.ylabel("Logistic cost function")
plt.savefig('images/03_logistic_regression_gd/AdalineGD_with_standardization.png')


# ------------------------------------------------------------------------------------------------------------------------------------------
# 4. VISUALIZE THE DECISION BOUNDARIES AND VERIFY THAT THE TRAINING SAMPLE IS CLASSIFIED CORRECTLY
# ------------------------------------------------------------------------------------------------------------------------------------------


# Function to visualize the decision boundaries

def plot_decision_regions(X, y, classifier, resolution=0.02):

    """ Create a colormap.

        Generate a matrix with two columns, where rows are all possible combinations of all numbers from min-1 to max+1 of the two series of
        features. The matrix with two columns is needed because the adaline was trained on a matrix with such shape.

        Use the predict method of the chosen classifier (ada) to predict the class corresponding to all the possible combinations of features
        generated in the above matrix. The predict method will use the weights learnt during the training phase: since the number of mis-
        classifications converged (even though to a non-zero value) in the training phase, we expect the perceptron to correctly classify all
        possible combinations of features.

        Reshape the vector of predictions as the X0_grid.

        Draw filled contours, where all possible combinations of features are associated to a Z, which is +1 or -1.

        To verify that the adaline correctly classified all possible combinations of the features, plot the the original features in the
        scatter plot and verify that they fall inside the correct region.
    """

    colors = ('red', 'blue', 'green')
    cmap = clr.ListedColormap(colors[:len(np.unique(y))])

    X0_min, X0_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    X1_min, X1_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    X0_grid, X1_grid = np.meshgrid(np.arange(X0_min, X0_max, resolution), np.arange(X1_min, X1_max, resolution))
    X0X1_combs = np.array([X0_grid.ravel(), X1_grid.ravel()]).T

    Z = classifier.predict(X0X1_combs)

    Z = Z.reshape(X0_grid.shape)

    plt.figure()
    plt.contourf(X0_grid, X1_grid, Z, alpha=0.3, cmap=cmap)
    plt.xlim(X0_min, X0_max)
    plt.ylim(X1_min, X1_max)

    for pos, cl in enumerate(np.unique(y)):

        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, color=colors[pos], marker='+', label=cl)


# Plot the decision region and the data

plot_decision_regions(X_train, y_train, classifier=lrgd)
plt.title('Decision boundary and training sample')
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.legend(loc='upper left')
plt.savefig('images/03_logistic_regression_gd/Decision_boundary_and_training_sample.png')


# ------------------------------------------------------------------------------------------------------------------------------------------
# 5. GENERAL
# ------------------------------------------------------------------------------------------------------------------------------------------


# Show plots

plt.show()


