""" adaline_gd
    ----------
    Implementation of a single layer adaptive linear neuron (with standardization) via gradient descent algorithm.
"""

# IMPORT LIBRARIES AND/OR MODULES


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as clr


# 1. DESIGN THE ADALINE

class AdalineGD(object):

    """ ADAptive LInear NEuron classifier

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
    cost_fun : list
        Sum of squares cost function value in each epoch.
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
        self.w = rgen.normal(loc=0.0, scale=0.01, size=[1 + X.shape[1], 1])
        self.cost_fun = []

        for epoch in range(self.n_epoch):

            update = y - self.activation(self.net_input(X))
            self.w[0] += self.eta * np.sum(update)
            self.w[1:] += self.eta * np.dot(X.T, update)
            cost = 0.5 * np.sum(update**2)
            self.cost_fun.append(cost)

        return self

    def net_input(self, X):

        """ Return the net input """

        net_input = (np.dot(X, self.w[1:]) + self.w[0])

        return net_input

    def activation(self, z):

        """ Return the linear activation """

        return z

    def predict(self, X):

        prediction = np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

        return prediction


# 2. PREPARE THE DATA


# Import the data

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
print(data.head())


# Extract the class labels

y = data.iloc[0:100, [4]].to_numpy()
y = np.where(y == "Iris-setosa", -1, 1)


# Extract the features

X = data.iloc[0:100, [0, 2]].to_numpy()


# Apply the standardization to scale the features (it can be verified that the adaline does not converge without standardization)

X_std = (X - np.mean(X, axis=0).reshape([1, 2])) / (np.std(X, axis=0).reshape([1, 2]))


# Plot the features in a scatter plot

plt.figure()
plt.scatter(X_std[:50, 0], X_std[:50, 1], color="red", marker="o", label="Setosa")
plt.scatter(X_std[50:100, 0], X_std[50:100, 1], color="blue", marker="x", label="Versicolor")
plt.xlabel("Sepal length [standardized]")
plt.ylabel("Petal length [standardized]")
plt.legend(loc="upper left")


# 3. TRAIN THE ADALINE


# Initialize the adaline object

ada = AdalineGD(eta=0.01, n_epoch=15)


# Learn from the data via the fit method (the activation method, rather than predict method, is called in the fit method to learn the weights)

ada.fit(X_std, y)


# Plot the cost function per epoch

plt.figure()
plt.plot(range(1, len(ada.cost_fun) + 1), (ada.cost_fun), marker="o")
plt.xlabel("Epochs")
plt.ylabel("log(Sum of squared errors)")
plt.title("AdalineGD with standard. - eta = 0.01")


# 4. VISUALIZE THE DECISION BOUNDARIES AND VERIFY THAT THE TRAINING SAMPLE IS CLASSIFIED CORRECTLY


# Function to visualize the decision boundaries

def plot_decision_regions(X, y, classifier, resolution=0.02):

    """ Create a colormap.

        Generate a matrix with two columns, where rows are all possible combinations of all numbers from min-1 to max+1 of the two series of
        features. The matrix with two columns is needed because the adaline was trained on a matrix with such shape.

        Use the predict method of the chosen classifier (ada) to predict the class corresponding to all the possible combinations of features
        generated in the above matrix. The predict method will use the weights learnt during the training phase: since the number of mis-
        classifications converged (even though to a non-zero value) in the training phase, we expect the perceptron to correctly classify all
        possible combinations of features.

        Reshape the vector of predictions as the x0_grid.

        Draw filled contours, where all possible combinations of features are associated to a Z, which is +1 or -1.

        To verify that the adaline correctly classified all possible combinations of the features, plot the the original features in the
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

    plt.scatter(X_std[:50, 0], X_std[:50, 1], alpha=0.8, color='red', marker='o', label='+1', edgecolor='black')
    plt.scatter(X_std[50:100, 0], X_std[50:100, 1], color='blue', marker='x', label='-1', edgecolor='black')
    plt.xlabel('Sepal length [standardized]')
    plt.ylabel('Petal length [standardized]')
    plt.legend(loc='upper left')


# Plot the decision region and the data

plot_decision_regions(X_std, y, classifier=ada)


# 5. GENERAL


# Show plots

plt.show()