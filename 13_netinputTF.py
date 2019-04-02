""" netinputTF
    ---------
    Implementation of a net input z of a sample point x in a one dimensional dataset with weights w and bias b via low
    level TensorFlow API.

"""


# IMPORT LIBRARIES AND/OR MODULES

import tensorflow as tf


# GRAPH AND SESSION

# Building a computational graph

g = tf.Graph()

with g.as_default():
    x = tf.placeholder(dtype=tf.float32, shape=(None), name="x")
    w = tf.Variable(2.0, name="weight")
    b = tf.Variable(0.7, name="basis")
    init = tf.global_variables_initializer()

    z = w*x + b


# Running the computational graph

with tf.Session(graph=g) as sess:

    # Initialize the variables w and b
    sess.run(init)

    # Evaluate z
    for t in [1.0, 0.6, -1.8]:
        print("X = {:>5,.2f} --> z = {:>5,.2f}".format(t, sess.run(z, feed_dict={x: t})))
