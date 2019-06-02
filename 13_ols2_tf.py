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


class TfLinreg:

    def __init__(self, X_train, y_train, learning_rate=0.01, random_seed=1, num_epochs=10):

        self.X_train = X_train
        self.y_train = y_train
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.num_epochs = num_epochs

        self.g = tf.Graph()

        with self.g.as_default():

            tf.set_random_seed(self.random_seed)

            self.X = tf.placeholder(dtype=tf.float32, shape=self.X_train.shape, name='X_input')
            self.y = tf.placeholder(dtype=tf.float32, shape=y_train.shape, name='y_input')

            w = tf.Variable(tf.zeros(shape=1), name='weight')
            b = tf.Variable(tf.zeros(shape=1), name='bias')

            self.z_net = tf.squeeze(w * self.X + b, name='z_net')

            sqr_errors = tf.square(self.y - self.z_net, name='sqr_errors')
            self.mean_cost = tf.reduce_mean(sqr_errors, name='mean_cost')

            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate, name='gradient_descent')
            self.optimizer = optimizer.minimize(self.mean_cost)

            self.init = tf.global_variables_initializer()

            self.sess = tf.Session(graph=self.g)

            self.sess.run(self.init)

    def train_linreg(self):

        training_cost = []

        for i in range(self.num_epochs):

            mean_cost, optimizer = self.sess.run([self.mean_cost, self.optimizer], feed_dict={self.X: self.X_train, self.y: self.y_train})
            training_cost.append(mean_cost)

        return training_cost

    def predict_linreg(self):

        y_pred = self.sess.run(self.z_net, feed_dict={self.X: self.X_train})

        return y_pred


# ------------------------------------------------------------------------------------------------------------------------------------------
# 2. PREPARE THE DATA
# ------------------------------------------------------------------------------------------------------------------------------------------


X_train = np.arange(10).reshape((10, 1))
y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0, 6.3, 6.6, 7.4, 8.0, 9.0])


# ------------------------------------------------------------------------------------------------------------------------------------------
# 3. TRAIN AND FORECAST
# ------------------------------------------------------------------------------------------------------------------------------------------


# Initialize the OLS

lr_model = TfLinreg(X_train, y_train)


# Train the model

training_costs = lr_model.train_linreg()


# Plot the cost

plt.figure()
plt.plot(range(1, len(training_costs) + 1), training_costs)
plt.xlabel('Epoch')
plt.ylabel('Training Cost')
plt.tight_layout()


# Predict using the trained model

y_predicted = lr_model.predict_linreg()


# Plot the best fit line

plt.figure()
plt.scatter(X_train, y_train, marker='o', label='Training Data')
plt.plot(range(X_train.shape[0]), y_predicted, color='gray', marker='s', markersize=6, linewidth=3, label='LinReg Model')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.tight_layout()


# ------------------------------------------------------------------------------------------------------------------------------------------
# 4. GENERAL
# ------------------------------------------------------------------------------------------------------------------------------------------


# Show plots

plt.show()