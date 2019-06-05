""" multilayer_perceptron_tf_k
    --------------------------
    Implementation of a multilayer perceptron via Tensorflow's keras.
"""


# ------------------------------------------------------------------------------------------------------------------------------------------
# 0. IMPORT LIBRARIES AND/OR MODULES
# ------------------------------------------------------------------------------------------------------------------------------------------


import os
import numpy as np
import struct
import tensorflow as tf
import tensorflow.keras as keras


# ------------------------------------------------------------------------------------------------------------------------------------------
# 1. PREPARE THE DATA
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

X_train, y_train = load_mnist('./', kind='train')
X_test, y_test = load_mnist('./', kind='t10k')


# Standardize the data

X_train_std = (X_train - np.mean(X_train, axis=0)) / np.std(X_train)
X_test_std = (X_test - np.mean(X_train, axis=0)) / np.std(X_train)

del X_train, X_test


# ------------------------------------------------------------------------------------------------------------------------------------------
# 2. DESIGN THE MULTILAYER PERCEPTRON
# ------------------------------------------------------------------------------------------------------------------------------------------


# Set random seeds

np.random.seed(123)
tf.set_random_seed(123)


# Design the network

y_train_onehot = keras.utils.to_categorical(y_train)

model = keras.models.Sequential()

model.add(keras.layers.Dense(units=50, input_dim=X_train_std.shape[1], kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='tanh'))
model.add(keras.layers.Dense(units=50, input_dim=50, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='tanh'))
model.add(keras.layers.Dense(units=y_train_onehot.shape[1], input_dim=50, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='softmax'))

sgd_optimizer = keras.optimizers.SGD(lr=0.001, decay=1e-7, momentum=.9)
model.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy')


# ------------------------------------------------------------------------------------------------------------------------------------------
# 3. TRAIN AND FORECAST
# ------------------------------------------------------------------------------------------------------------------------------------------


# Train the network

history = model.fit(X_train_std, y_train_onehot, batch_size=64, epochs=50, verbose=1, validation_split=0.1)


# Make predictions

y_train_pred = model.predict_classes(X_train_std, verbose=0)
correct_pred = np.sum(y_train == y_train_pred, axis=0)
train_acc = correct_pred / y_train.shape[0]

print('\nTraining accuracy: {:.2f}%'.format(train_acc*100))

y_test_pred = model.predict_classes(X_test_std, verbose=0)
correct_pred = np.sum(y_test == y_test_pred, axis=0)
test_acc = correct_pred / y_test.shape[0]

print('\nTest accuracy: {:.2f}%'.format(test_acc*100))
