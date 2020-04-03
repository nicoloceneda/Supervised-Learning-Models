""" LINEAR REGRESSION - TENSOR FLOW
    -------------------------------
    Implementation of a multilayer perceptron for multi-class classification, with two hidden layer, using tensorflow.
"""


# -------------------------------------------------------------------------------
# 0. IMPORT LIBRARIES
# -------------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds


# -------------------------------------------------------------------------------
# 1. PREPARE THE DATA
# -------------------------------------------------------------------------------


# Import the dataset

iris, iris_info = tfds.load('iris', with_info=True)


# Print the dataset information

print(iris_info)


# Separate the data into train and test subsets

tf.random.set_seed(1)
ds_orig = iris['train'].shuffle(150, reshuffle_each_iteration=False)
ds_train = ds_orig.take(100)
ds_test = ds_orig.skip(100)


# Create tuples of features and class labels

ds_train = ds_train.map(lambda item: (item['features'], item['label']))
ds_test = ds_test.map(lambda item: (item['features'], item['label']))

