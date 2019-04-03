""" olsTF
    ---------
    Implementation of a ordinary least squares regression via low level TensorFlow API.

"""


# IMPORT LIBRARIES AND/OR MODULES

import tensorflow as tf
import numpy as np
import matplotlib as plt


# DESIGN THE CLASS THAT IMPLEMENTS PREDICTIONS VIA OLS


class TfLinreg(object):

    """ Ordinary least squares

    Parameters:
    -----------
    x_coln : int
        Number of columns in the vector/matrix X of the independent variable.
    eta : float
        Learning rate (between 0.0 and 1.0).
    random_seed : int
        Graph level random seed.

    Attributes:
    -----------
    w_ : 1d-array
        Weights after fitting.
    n_miscl_ : list
        Number of n_misclassifications (updates) in each epoch.

    """

    """ Two placeholders: x and y

        Two trainable variables: w and b

        Define the linear regression model: z = w * x +b

        Define the cost function: MSE

        Learn the weights via gradient descent algo

    """

    def __init__(self, x_coln, eta=0.01, random_seed=None):
        self.x_coln = x_coln
        self.eta = eta
        self.random_seed = random_seed

        # Building a computational graph
        self.g = tf.Graph()

        with self.g.as_default():

            # Set the graph-level random seed
            tf.set_random_seed(self.random_seed)

            # Call the build method
            self.build()

            # Initialize the global variables
            self.init = tf.global_variables_initializer()

    def build(self):

        # Import the dependent and independent variables
        self.X = tf.placeholder(dtype=tf.float32, shape=(None, self.x_coln), name="X")
        self.y = tf.placeholder(dtype=tf.float32, shape=None, name="y")
        print(self.X)
        print(self.y)

        # Initialize to zero the trainable variables w and b
        w = tf.Variable(tf.zeros(shape=1), name="weight")
        b = tf.Variable(tf.zeros(shape=1), name="bias")
        print(w)
        print(b)

        self.z_net = tf.squeeze(w*self.X + b, name="z_net")
        print(self.z_net)

        sqr_errors = tf.square(self.y - self.z_net, name="sqr_errors")
        print(sqr_errors)

        self.mean_cost = tf.reduce_mean(sqr_errors, name="mean_cost")

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.eta, name="GradientDescent")
        self.optimizer = optimizer.minimize(self.mean_cost)


# INSTANTIATION


X_train = np.arange(10).reshape(10, 1)
y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0, 6.3, 6.6, 7.4, 8.0, 9.0])

lrmodel = TfLinreg(x_coln=X_train.shape[1], eta=0.01)


# TRAINING


# Implement a training function to learn the weights of the linear regression model

def train_linreg(sess, model, X_train, y_train, num_epochs=10):

    # Initialize the global variables w and b
    sess.run(model.init_op)

    # Calculate the training cost
    training_costs = []

    for i in range(num_epochs):
        _, cost = sess.run([model.optimizer, model.mean_cost], feed_dict={model.X: X_train, model.y: y_train})
        training_costs.append(cost)

    return training_costs


# Session to launch the lrmodel.g graph and pass the required arguments to the train_linreg function for training

sess = tf.Session(graph=lrmodel.g)
training_costs = train_linreg(sess, lrmodel, X_train, y_train)


# Plot the training cost

plt.figure()
plt.plot(range(1, len(training_costs) +1), training_costs)
plt.tight_layout()
plt.xlabel("Epoch")
plt.ylabel("Training cost")


# PREDICTION


# Implement a function to make predictions based on the input features

def predict_linreg(sess, model, X_test):
    y_pred = sess.run(model.z_net, feed_dict={model.X: X_test})
    return y_pred

# Plot the linear regression fit on the training data

plt.scatter(X_train, y_train, marker='s', s=50, label='Training Data')
plt.plot(range(X_train.shape[0]), predict_linreg(sess, lrmodel, X_train), color='gray', marker='o', markersize=6,
         linewidth=3, label='LinReg Model')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.tight_layout()


# Show plots

plt.show()