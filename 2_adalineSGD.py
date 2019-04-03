""" adalineSGD
    ----------
    Implementation of a single layer Adaptive Linear Neuron via a stochastic gradient descent algorithm. A
    standardization is applied.

"""


# IMPORT LIBRARIES AND/OR MODULES


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as clr


# DESIGN THE ADALINE


class AdalineSGD(object):

    """ ADAptive LInear NEuron classifier

    Parameters:
    -----------
    eta : float
        Learning rate (between 0.0 and 1.0).
    n_epoch : int
        Passes over the training dataset.
    random_state : int
        Random number generator seed for random weight initialization.
    shuffle : bool (default: True)
        If set to True, it shuffles the training dataset every epoch to prevent cycles.

    Attributes:
    -----------
    w_ : 1d-array
        Weights after fitting.
    cost_ : list
        Sum of squares cost function value averaged over all training samples in each epoch.

    """

    def __init__(self, eta=0.01, n_epoch=10, random_state=1, shuffle=True):
        self.eta = eta
        self.n_epoch = n_epoch
        self.random_state = random_state
        self.shuffle = shuffle

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

        self._initialize_weights(X.shape[1])
        self.cost_ = []

        for i in range(self.n_epoch):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)

        return self

    def partial_fit(self, X, y):

        """ Fit training data without reinitializing the weights """

        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)

        return self

    def _shuffle(self, X, y):

        """ Shuffle training data """

        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):

        """ Initialize weights to small random numbers """

        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):

        """ Apply the Adaline learning rule to update the weights """

        net_input = self.net_input(xi)
        output = self.activation(net_input)
        error = (target - output)
        self.w_[0] += self.eta*error
        self.w_[1:] += self.eta*xi.dot(error)
        cost = (error**2) / 2.0
        return cost

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


# Apply standardization for feature scaling

X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()


# Initialize an ADELINE object

ada = AdalineSGD(n_epoch=15, eta=0.01)


# Learn from data via the fit method

ada.fit(X_std, y)


# Plot the cost per epoch

plt.figure()
plt.plot(range(1, len(ada.cost_) +1), ada.cost_, marker="o")
plt.xlabel("Epochs")
plt.ylabel("Average cost")
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
