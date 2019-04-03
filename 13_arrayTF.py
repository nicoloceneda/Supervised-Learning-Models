""" arrayTF
    -------
    Creation of a simple rank-3 tensor of size batchsize x2 x3, reshaping of it, and calculation of the column sums and
    mean via low level TensorFlow API.

"""


# IMPORT LIBRARIES AND/OR MODULES


import tensorflow as tf
import numpy as np


# BUILDING THE COMPUTATIONAL GRAPH


g = tf.Graph()

with g.as_default():

    # Import the input array
    X = tf.placeholder(dtype=tf.float32, shape=(None, 2, 3), name="X")

    # Reshape the array
    X_res = tf.reshape(X, shape=(-1, 6), name="reshaped_X")

    # Calculate the sum and the mean of the reshaped array
    X_sum = tf.reduce_sum(X_res, axis=0, name="col_sum")
    X_mean = tf.reduce_mean(X_res, axis=0, name="col_mean")


# RUNNING THE COMPUTATIONAL GRAPH


with tf.Session(graph=g) as sess:

    # Generate a random array
    X_array = np.arange(18).reshape(3, 2, 3)

    # Evaluating the initial shape, the reshaped array and the mean and sum of its columns
    print("Shape input array: ", X_array.shape)
    print("Reshaped array:\n", sess.run(X_res, feed_dict={X: X_array}))
    print("Columns sums:\n", sess.run(X_sum, feed_dict={X: X_array}))
    print("Columns means:\n", sess.run(X_mean, feed_dict={X: X_array}))

