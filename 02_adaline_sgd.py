""" adaline_sgd
    -----------
    Implementation of a single layer adaptive linear neuron (with standardization) via stochastic gradient descent algorithm.
"""


# ------------------------------------------------------------------------------------------------------------------------------------------
# 0. IMPORT LIBRARIES AND/OR MODULES
# ------------------------------------------------------------------------------------------------------------------------------------------


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as clr


# ------------------------------------------------------------------------------------------------------------------------------------------
# 1. IMPLEMENT THE ADALINE
# ------------------------------------------------------------------------------------------------------------------------------------------


class AdalineSGD(object):

    """ ADAptive LInear NEuron classifier

    Parameters:
    -----------
    eta : float
        Learning rate (between 0.0 and 1.0).
    n_iter : int
        Passes over the training dataset.
    random_state : int
        Random number generator seed for random weight initialization.
    shuffle : bool (default: True)
        If set to True, it shuffles the training dataset every iter to prevent cycles.

    Attributes:
    -----------
    w : 1d-array
        Weights after fitting.
    cost_fun : list
        Sum of squares cost function value averaged over all training samples in each iter.
    """

    def __init__(self, eta=0.01, n_iter=10, random_state=1, shuffle=True):

        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.shuffle = shuffle

    def fit(self, X, y):

        """ Fit training data

        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and n_features is the number of features.
        Y : array-like shape = [n_samples, ]
            Target values.

        Returns:
        --------
        self : object
        """

        rgen = np.random.RandomState(self.random_state)
        self.w = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.avg_cost_fun = []

        for iteration in range(self.n_iter):

            if self.shuffle:

                X, y = self.shuffler(X, y)

            cost = []
            # cost = 0

            for Xi, yi in zip(X, y):

                cost.append(self.update_weights(Xi, yi))
                # cost += self.update_weights(Xi, yi)

            self.avg_cost_fun.append(sum(cost) / len(cost))
            # self.avg_cost_fun.append(sum(cost) / len(y))

        return self

    def update_weights(self, Xi, yi):

        """ Apply Adaline learning rule to update weights. """

        update = yi - self.activation(self.net_input(Xi))
        self.w[0] += self.eta * update
        self.w[1:] += self.eta * np.dot(Xi, update)
        cost = 0.5 * (update ** 2)

        return cost

    def shuffler(self, X, y):

        """ Shuffle training data. """

        rgen = np.random.RandomState(self.random_state)
        pos = rgen.permutation(len(y))

        return X[pos], y[pos]

    def net_input(self, Xi):

        net_input = np.dot(Xi, self.w[1:].T) + self.w[0]

        return net_input

    def activation(self, z):

        """ Return the linear activation """

        return z

    def predict(self, X):

        """ Return class label after unit step. """

        prediction = np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

        return prediction


# ------------------------------------------------------------------------------------------------------------------------------------------
# 2. PREPARE THE DATA
# ------------------------------------------------------------------------------------------------------------------------------------------


# Import the data

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
print(data.head())


# Extract the class labels

y = data.iloc[0:100, 4].to_numpy()
y = np.where(y == "Iris-setosa", -1, 1)


# Extract the features

X = data.iloc[0:100, [0, 2]].to_numpy()


# Apply the standardization to scale the features (it can be verified that the adaline does not converge without standardization)

X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)


# Plot the features in a scatter plot

plt.figure()
plt.scatter(X_std[:50, 0], X_std[:50, 1], color="red", marker="o", label="Setosa")
plt.scatter(X_std[50:100, 0], X_std[50:100, 1], color="blue", marker="x", label="Versicolor")
plt.title("Scatter plot of the scaled features")
plt.xlabel("Sepal length [standardized]")
plt.ylabel("Petal length [standardized]")
plt.legend(loc="upper left")


# ------------------------------------------------------------------------------------------------------------------------------------------
# 3. TRAIN THE ADALINE
# ------------------------------------------------------------------------------------------------------------------------------------------


# Initialize the adaline object

ada = AdalineSGD(eta=0.01, n_iter=15)


# Learn from the data via the fit method (the activation method, rather than predict method, is called in the fit method to learn the weights)

ada.fit(X_std, y)


# Plot the cost function per iter

plt.figure()
plt.plot(range(1, len(ada.avg_cost_fun) + 1), ada.avg_cost_fun, marker="o")
plt.title("AdalineSGD with standard")
plt.xlabel("iters")
plt.ylabel("Average Sum of squared errors")


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

    cmap = clr.ListedColormap(['red', 'blue'])

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

    plt.scatter(X_std[:50, 0], X_std[:50, 1], alpha=0.8, color='red', marker='+', label='+1')
    plt.scatter(X_std[50:100, 0], X_std[50:100, 1], color='blue', marker='+', label='-1')
    plt.title('Decision boundary and training sample')
    plt.xlabel('Sepal length [standardized]')
    plt.ylabel('Petal length [standardized]')
    plt.legend(loc='upper left')


# Plot the decision region and the data

plot_decision_regions(X_std, y, classifier=ada)


# ------------------------------------------------------------------------------------------------------------------------------------------
# 5. GENERAL
# ------------------------------------------------------------------------------------------------------------------------------------------


# Show plots

plt.show()