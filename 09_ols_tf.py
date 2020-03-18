
# -------------------------------------------------------------------------------
# 0. IMPORT LIBRARIES
# -------------------------------------------------------------------------------


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------------
# 2. PREPARE THE DATA
# -------------------------------------------------------------------------------


# Generate the dataset

X_train = np.arange(10).reshape((10, 1))
y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0, 6.3, 6.6, 7.4, 8.0, 9.0])


# Plot the dataset

plt.plot(X_train, y_train, 'o',)
plt.xlabel('x')
plt.ylabel('y')


# Standardize the features

X_train_norm = (X_train - np.mean(X_train)) / np.std(X_train)


# Create a dataset

ds_train_orig = tf.data.Dataset.from_tensor_slices((tf.cast(X_train_norm, tf.float32), tf.cast(y_train, tf.float32)))


# Define a model for linear regression

class OrdinaryLeastSquares(tf.keras.Model):

    def __init__(self):



# -------------------------------------------------------------------------------
# 5. GENERAL
# -------------------------------------------------------------------------------


# Show plots

plt.show()
