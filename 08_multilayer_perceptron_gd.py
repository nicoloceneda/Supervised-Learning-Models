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

    def one_hot_representation(self, y, n_classes):

        """ Encode
        """

