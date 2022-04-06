""" LOGISTIC REGRESSION - GRADIENT DESCENT
    --------------------------------------
    Implementation of a single layer logistic regression for binary classification, via gradient descent algorithm, with standardized
    features.
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

y = data.iloc[:100, 4].to_numpy()
y = np.where(y == 'Iris-setosa', 0, 1)


# Extract the features

X = data.iloc[:100, [0, 2]].to_numpy()


# Apply the standardization to scale the features

X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)


# Plot the features in a scatter plot

plt.figure()
plt.scatter(X_std[:50, 0], X_std[:50, 1], color="red", marker="+", label="Setosa")
plt.scatter(X_std[50:, 0], X_std[50:, 1], color="blue", marker="+", label="Versicolor")
plt.title("Scatter plot of the scaled features")
plt.xlabel("Sepal length [standardized]")
plt.ylabel("Petal length [standardized]")
plt.legend(loc="upper left")
plt.savefig('images/03_logistic_regression_gd/Scatter_plot_of_the_scaled_features.png')


# -------------------------------------------------------------------------------
# 2. DESIGN THE MODEL
# -------------------------------------------------------------------------------


# Design the logistic regression

class LogisticRegressionGD:

    """ Logistic regression classifier

        Parameters:
        ----------
        eta : float
            Learning rate (between 0.0 and 1.0)
        n_epochs : int
            Number of epochs.

        Attributes:
        ----------
        w : array, shape = [n_features+1, ]
            Weights after fitting.
        cost_fun : list
            Logistic cost function value in each epoch.
    """

    def __init__(self, eta=0.01, n_epochs=100):

        self.eta = eta
        self.n_epochs = n_epochs

    def fit(self, X, y):

        """ Fit training set

            Parameters:
            ----------
            X : array, shape = [n_samples, n_features]
            y : array, shape = [n_samples, ]

            Returns:
            --------
            self : object
        """

        rgen = np.random.RandomState(seed=1)
        self.w = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_fun = []

        for epoch in range(self.n_epochs):

            phi_z = self.sigmoid_activ(X)
            update = y - phi_z
            self.w[0] += self.eta * np.sum(update)
            self.w[1:] += self.eta * np.dot(X.T, update)
            cost = -np.dot(y, np.log(phi_z)) - np.dot((1 - y), (1 - phi_z))
            self.cost_fun.append(cost)

        return self

    def sigmoid_activ(self, X):

        """ Calculate the net input and return the probability level after the logistic sigmoid function
            (Used in the fit method)

            Parameters:
            ----------
            X : array, shape = [n_samples, n_features]

            Returns:
            --------
            sigmoid_activ : array, shape = [n_samples, ]
        """

        net_input = self.w[0] + np.dot(X, self.w[1:])

        return 1 / (1 + np.exp(-np.clip(net_input, -250, 250)))

    def step_activ(self, X):

        """ Calculate the net input and return the class label prediction after the unit step function
            (Used in plot_decision_regions function)

            Parameters:
            ----------
            X : array, shape = [X0X1_combs.shape[0], ]

            Returns:
            -------
            step_activ : array, shape = [X0X1_combs.shape[0], ]
        """

        net_input = self.w[0] + np.dot(X, self.w[1:])

        return np.where(net_input >= 0, 1, 0)


# -------------------------------------------------------------------------------
# 3. TRAIN THE MODEL
# -------------------------------------------------------------------------------


# Initialize a logistic regression object

logreg = LogisticRegressionGD(eta=0.05, n_epochs=1000)


# Learn from the data via the fit method

logreg.fit(X_std, y)


# Plot the cost function per epoch

plt.figure()
plt.plot(range(1, len(logreg.cost_fun) + 1), logreg.cost_fun, marker='o')
plt.title('Cost function per epoch')
plt.xlabel('Epoch')
plt.ylabel('Cost function')
plt.savefig('images/03_logistic_regression_gd/LogisticRegressionGD_with_standardization.png')


# -------------------------------------------------------------------------------
# 4. EVALUATE THE MODEL
# -------------------------------------------------------------------------------

# Function to plot the decision boundary

def plot_decision_regions(X, y, classifier, resolution=0.02):

    """ Create a colormap object.

        Generate a matrix with two columns, where rows are all possible combinations of all numbers from min-1 to max+1 of the two series of
        features. The matrix with two columns is needed because the logistic regression was trained on a matrix with such shape.

        Use the step_activ method of the logreg to predict the class corresponding to all the possible combinations of features generated in
        the above matrix. The step_activ method will use the weights learnt during the training phase: since the cost function converged du-
        ring the training phase, we expect the logreg to find a decision boundary that correctly classifies all the samples in the training
        set.

        Reshape the vector of predictions as the X0_grid.

        Draw filled contours, where all possible combinations of features are associated to a Z, which is 1 or 0.

        To verify that the adaline correctly classified all the samples in the training set, plot the the original features in the scatter
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

plot_decision_regions(X_std, y, classifier=logreg)
plt.title('Decision boundary and training sample')
plt.xlabel('Sepal length [standardized]')
plt.ylabel('Petal length [standardized]')
plt.legend(loc='upper left')
plt.savefig('images/03_logistic_regression_gd/Decision_boundary_and_training_sample.png')


# -------------------------------------------------------------------------------
# 5. GENERAL
# -------------------------------------------------------------------------------


# Show plots

plt.show()