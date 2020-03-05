""" SUPPORT VECTOR MACHINE - GRADIENT DESCENT - SCIKIT LEARN
    --------------------------------------------------------
    Implementation of a support vector machine for multi-class classification, via gradient descent algorithm, with standardized features,
    using scikit-learn.
"""


# -------------------------------------------------------------------------------
# 0. IMPORT LIBRARIES
# -------------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from sklearn import datasets, model_selection, preprocessing, svm, metrics


# -------------------------------------------------------------------------------
# 1. PREPARE THE DATA
# -------------------------------------------------------------------------------


# Import the dataset

data = datasets.load_iris()


# Extract the class labels

y = data.target




