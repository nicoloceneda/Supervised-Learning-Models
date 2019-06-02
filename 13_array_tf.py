import tensorflow as tf
import numpy as np

g = tf.Graph()

with g.as_default():

    x_input = tf.placeholder(dtype=tf.float32, shape=(None, 2, 3), name='x_input')

    x_reshaped = tf.reshape(x_input, shape=(-1, 6), name='x_reshaped')
    col_sum = tf.reduce_sum(x_reshaped, axis=0, name='col_sum')
    col_mean = tf.reduce_mean(x_reshaped, axis=0, name='col_mean')

with tf.Session(graph=g) as sess:

    x_array = np.arange(18).reshape(3, 2, 3)

    print('Input shape:\n', x_array.shape)
    print('Reshaped array:\n', sess.run(x_reshaped, feed_dict={x_input: x_array}))
    print('Column sums:\n', sess.run(col_sum, feed_dict={x_input: x_array}))
    print('Column means:\n', sess.run(col_mean, feed_dict={x_input: x_array}))