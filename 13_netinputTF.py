""" netinputTF
    ---------
    Implementation of a net input z of a sample point x in a one dimensional dataset with weights w and bias b via low
    level TensorFlow API.

"""


# IMPORT LIBRARIES AND/OR MODULES


import tensorflow as tf


# BUILDING THE COMPUTATIONAL GRAPH


g = tf.Graph()

with g.as_default():

    # Import the input x
    x = tf.placeholder(dtype=tf.float32, shape=None, name="x")

    # Initialize the global variables w and b
    w = tf.Variable(2.0, name="weight")
    b = tf.Variable(0.7, name="bias")
    init = tf.global_variables_initializer()

    # Calculate the net input
    z = w*x + b


# RUNNING THE COMPUTATIONAL GRAPH


with tf.Session(graph=g) as sess:

    # Initialize the variables w and b
    sess.run(init)

    # Evaluate z at each x
    for i in [1.0, 0.6, -1.8]:
        print("X = {:>5,.2f} --> z = {:>5,.2f}".format(i, sess.run(z, feed_dict={x: i})))
