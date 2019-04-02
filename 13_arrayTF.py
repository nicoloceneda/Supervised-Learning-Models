""" arrayTF
    -------
    Creation of a simple rank-3 tensor of size batchsize x2 x3, reshaping of it, and calculation of the column sums and
    mean via low level TensorFlow API.

"""


# IMPORT LIBRARIES AND/OR MODULES

import tensorflow as tf
import numpy as np


# GRAPH AND SESSION

# Building a computational graph

g = tf.Graph()

with g.as_default():
    x = tf.placeholder(dtype=tf.float32, shape=(None, 2, 3), name="X")

    x_res = tf.reshape(x, shape=(-1, 6), name="reshaped")
    x_sum = tf.reduce_sum(x_res, axis=0, name="col_sum")
    x_mean = tf.reduce_mean(x_res, axis=0, name="col_mean")


# Running the computational graph

with tf.Session(graph=g) as sess:
    x_array = np.arange(18).reshape(3, 2, 3)
    print("Input shape: ", x_array.shape)
    print("Reshaped:\n", sess.run(x_res, feed_dict={x: x_array}))
    print("Columns sums:\n", sess.run(x_sum, feed_dict={x: x_array}))
    print("Columns means:\n", sess.run(x_mean, feed_dict={x: x_array}))

