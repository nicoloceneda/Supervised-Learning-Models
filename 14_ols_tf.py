""" ols
    ---
    Implementation of an OLS via Tensorflow.
"""


# ------------------------------------------------------------------------------------------------------------------------------------------
# 0. IMPORT LIBRARIES AND/OR MODULES
# ------------------------------------------------------------------------------------------------------------------------------------------


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------------------------------------------------------------------
# 1. DESIGN THE OLS
# ------------------------------------------------------------------------------------------------------------------------------------------


# Set random seeds

np.random.seed(0)
tf.set_random_seed(123)


# Design the OLS

g = tf.Graph()

with g.as_default():

    # Placeholders

    tf_x = tf.placeholder(shape=None, dtype=tf.float32, name='tf_x')
    tf_y = tf.placeholder(shape=None, dtype=tf.float32, name='tf_y')

    # Variables

    weight = tf.Variable(tf.random_normal(shape=(1, 1), stddev=0.25), name='weight')
    bias = tf.Variable(0.0, name='bias')

    init = tf.global_variables_initializer()

    # Prediction

    y_hat = tf.add(weight * tf_x, bias, name='y_hat')

    # Cost

    cost = tf.reduce_mean(tf.square(tf_y - y_hat), name='cost')

    # Optimizer

    optim = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optim.minimize(cost, name='train_op')


# ------------------------------------------------------------------------------------------------------------------------------------------
# 2. PREPARE THE DATA
# ------------------------------------------------------------------------------------------------------------------------------------------


# Generate random data

def make_random_data():

    x = np.random.uniform(low=-2, high=4, size=200)
    y = [np.random.normal(loc=0.0, scale=(0.5 + t * t / 3), size=None) for t in x]

    return x, 1.726 * x - 0.84 + np.array(y)


x, y = make_random_data()


# Plot the random data generated

plt.figure()
plt.plot(x, y, 'o')


# ------------------------------------------------------------------------------------------------------------------------------------------
# 3. TRAIN
# ------------------------------------------------------------------------------------------------------------------------------------------

x_train, y_train = x[:100], y[:100]
x_test, y_test = x[100:], y[100:]

n_epochs = 500
training_costs = []

with tf.Session(graph=g) as sess:

    sess.run(init)

    for epoch in range(n_epochs):

        c, _ = sess.run([cost, train_op], feed_dict={tf_x: x_train, tf_y: y_train})
        training_costs.append(c)

        print('Epoch {:3}: {:7.4f}'.format(epoch, c))


# Plot the cost

plt.figure()
plt.plot(training_costs)
plt.show()