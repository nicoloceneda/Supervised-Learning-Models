""" MULTILAYER PERCEPTRON - GRADIENT DESCENT
    ----------------------------------------
    Implementation of a multilayer perceptron for multi-class classification, with one hidden layer.
"""


# -------------------------------------------------------------------------------
# 0. IMPORT LIBRARIES
# -------------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------------
# 1. DESIGN THE MULTILAYER PERCEPTRON
# -------------------------------------------------------------------------------


# Import the dataset

mnist = np.load('mnist dataset/compressed/mnist_std.npz')

# Extract the class labels

y_train = mnist['y_train']
y_test = mnist['y_test']


# Extract the features

X_train_std = mnist['X_train_std']
X_test_std = mnist['X_test_std']


# -------------------------------------------------------------------------------
# 2. TRAIN THE PERCEPTRON
# -------------------------------------------------------------------------------


class MultilayerPerceptron:

    """ Multilayer Perceptron classifier

        Parameters:
        ----------
        eta : float
            Learning rate (between 0.0 and 1.0)
        n_epochs : int
            Number of epochs.
        shuffle : bool
            If set to true it shuffles the training set before each epoch to prevent cycles.
        n_samples_mb : int
            Number of training samples per minibatch.
        n_units_h : int
            Number of units in the hidden layer.
        l2 : float
            Lambda value for L2-regularization.

        Attributes:
        ----------
        eval_train : dict
            Dictionary containing the cost, training accuracy and validation accuracy for each epoch during training.
    """

    def __init__(self, eta=0.01, n_epochs=100, shuffle=True, n_samples_mb=1, n_units_h=30, l2=0.0):

        self.eta = eta
        self.n_epochs = n_epochs
        self.shuffle = shuffle
        self.n_samples_mb = n_samples_mb
        self.n_units_h = n_units_h
        self.l2 = l2

    def one_hot_encoding(self, y_train):

        """ Encode the labels into the one-hot representation
            (Used in fit method)

            Parameters:
            ----------
            y_train : array, shape = [n_samples, ]

            Returns:
            -------
            onehot : array, shape = [n_samples, n_labels]
        """

        onehot = np.zeros(y_train.shape[0], np.unique(y_train).shape[0])

        for sample, label in enumerate(y_train.astype(int)):

            onehot[sample, label] = 1

        return onehot

    def sigmoid_activ(self, Z):

        """ Return the probability level after the logistic sigmoid function
            (Used in forward propagate method)

            Parameters:
            ----------
            Z : array, shape = [n_samples_mb, n_units_h-1]

            Returns:
            -------
            sigmoid_active : array
        """

        return 1 / (1 + np.exp(-np.clip(net_input, -250, 250)))

    def forward_propagate(self, A_in):

        """ Compute the forward propagation step
            (Used in :TODO)

            Parameters:
            ----------
            A_in :

            Returns:
            -------

        """

        Z_h = self.b_h + np.dot(A_in, self.W_h)
        A_h = self.sigmoid_activ(Z_h)

        Z_out = self.b_out + np.dot(A_h, self.W_out)
        A_out = self.sigmoid_activ(Z_out)

        return Z_h, A_h, Z_out, A_out

    def compute_cost(self, y_enc, output):

        """ Compute cost function

            Parameters:
            ----------
            y_enc : array, shape = [n_samples, n_labels]
            output : array, shape = [n_samples, n_output_nits]

            Returns:
            -------
            cost : float
        """

        l2_term = self.l2 * (np.sum(self.W_h ** 2) + np.sum(self.W_out ** 2))
        cost = np.sum(- y_enc * np.log(output) - (1 - y_enc) * np.log(1 - output)) + l2_term # TODO: why not -l2_term

    def predict(self, X):

        """ Predict class labels

            Parameters:
            ----------
            X : array, shape = [n_samples, n_features]

            Returns:
            -------
            y_pred : array, shape = [n_samples, ]
        """

        Z_h, A_h, Z_out, A_out = self.forward(X)
        y_pred = np.argmax(Z_out, axis=1)
        return y_pred

    def fit(self, X_train, y_train, X_valid, y_valid):

        """ Learn the weights from the training data

            Parameters:
            ----------
            X_train : array, shape = [n_samples_train, n_features]
            y_train : array, shape = [n_samples_train, ]
            X_valid : array, shape = [n_samples_valid, n_features]
            y_valid : array, shape = [n_samples_valid, ]

            Returns:
            -------
            self : object
        """

        n_features = X_train.shape[1]
        n_labels = np.unique(y_train).shape[0]

        rgen = np.random.RandomState(seed=1)

        self.b_h = np.zeros(self.n_units_h)
        self.W_h = rgen.normal(loc=0.0, scale=0.1, size=(n_features, n_labels))
        self.b_out = np.zeros(n_labels)
        self.W_out = rgen.normal(loc=0.0, scale=0.1, size=(self.n_units_h, n_labels))


# -------------------------------------------------------------------------------
# 3. TRAIN THE MULTILAYER PERCEPTRON
# -------------------------------------------------------------------------------


# Initiate a multilayer perceptron object

mlp = MultilayerPerceptron(eta=0.0005, n_epochs=200, shuffle=True, n_samples_mb=100, n_units_h=100, l2=0.01)


# Learn from the data via the fit method

mlp.fit(X_train_std[:55000], y_train[:55000], X_train_std[55000:], y_train[55000:])