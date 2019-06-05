""" multilayer_perceptron_tf_l
    --------------------------
    Implementation of a multilayer perceptron via Tensorflow's Layers.
"""


# ------------------------------------------------------------------------------------------------------------------------------------------
# 0. IMPORT LIBRARIES AND/OR MODULES
# ------------------------------------------------------------------------------------------------------------------------------------------


import os
import numpy as np
import struct
import tensorflow as tf


# ------------------------------------------------------------------------------------------------------------------------------------------
# 2. PREPARE THE DATA
# ------------------------------------------------------------------------------------------------------------------------------------------


# Create a function to load the data

def load_mnist(path, kind='train'):

    """Load MNIST data from `path`"""

    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:

        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:

        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.) - .5) * 2

    return images, labels


# Load the data

X_train, y_train = load_mnist('.', kind='train')
X_test, y_test = load_mnist('.', kind='t10k')


# Standardize the data

X_train_std = (X_train - np.mean(X_train, axis=0)) / np.std(X_train)
X_test_std = (X_test - np.mean(X_train, axis=0)) / np.std(X_train)

del X_train, X_test


# ------------------------------------------------------------------------------------------------------------------------------------------
# 1. DESIGN THE MULTILAYER PERCEPTRON
# ------------------------------------------------------------------------------------------------------------------------------------------


# Set random seeds

np.random.seed(123)
tf.set_random_seed(123)


# Build the graph

n_features = X_train_std.shape[1]
n_classes = len(np.unique(y_train))

g = tf.Graph()

with g.as_default():

    tf_x = tf.placeholder(dtype=tf.float32, shape=(None, n_features), name='tf_x')
    tf_y = tf.placeholder(dtype=tf.int32, shape=None, name='tf_y')
    y_onehot = tf.one_hot(indices=tf_y, depth=n_classes)

    h1 = tf.layers.dense(inputs=tf_x, units=50, activation=tf.tanh, name='layer1')
    h2 = tf.layers.dense(inputs=h1, units=50, activation=tf.tanh, name='layer2')
    logits = tf.layers.dense(inputs=h2, units=10, activation=None, name='layer3')

    predictions = {'classes': tf.argmax(logits, axis=1, name='predicted_classes'), 'probabilities': tf.nn.softmax(logits, name='softmax_tensor')}

    cost = tf.losses.softmax_cross_entropy(onehot_labels=y_onehot, logits=logits)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss=cost)

    init = tf.global_variables_initializer()


# ------------------------------------------------------------------------------------------------------------------------------------------
# 2. TRAIN AND FORECAST
# ------------------------------------------------------------------------------------------------------------------------------------------


# Create batches

def create_batch_generator(X, y, batch_size=128, shuffle=False):

    X_copy = X.copy()
    y_copy = y.copy()

    if shuffle:

        data = np.column_stack((X_copy, y_copy))
        np.random.shuffle(data)
        X_copy = data[:, :-1]
        y_copy = data[:, -1]

    for i in range(0, X.shape[0], batch_size):

        yield (X_copy[i:i + batch_size, :], y_copy[i:i + batch_size])


# Train the network

sess = tf.Session(graph=g)

sess.run(init)

training_costs = []

for epoch in range(50):

    training_loss = []
    batch_generator = create_batch_generator(X=X_train_std, y=y_train, batch_size=64)

    for batch_X, batch_y in batch_generator:

        _, batch_cost = sess.run([train_op, cost], feed_dict={tf_x: batch_X, tf_y: batch_y})
        training_costs.append(batch_cost)

    print(' -- Epoch {:2} | Mean Training Loss: {:.4f}'.format(epoch + 1, np.mean(training_costs) ))


# Make predictions

y_pred = sess.run(predictions['classes'], feed_dict={tf_x: X_test_std})
print('Test accuracy: {}'.format(100*np.sum(y_pred == y_test) / y_test.shape[0]))