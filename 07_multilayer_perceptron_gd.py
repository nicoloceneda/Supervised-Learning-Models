""" MULTILAYER PERCEPTRON - GRADIENT DESCENT
    ----------------------------------------
    Implementation of a multilayer peRceptron for multi-class classification.
"""


# -------------------------------------------------------------------------------
# 0. IMPORT LIBRARIES
# -------------------------------------------------------------------------------


import os
import struct
import numpy as np


# -------------------------------------------------------------------------------
# 1. PREPARE THE DATA
# -------------------------------------------------------------------------------


# Function to import the mnist dataset

def load_mnist(path, kind):

    labels_path = os.path.join(path, '{}-labels.idx1-ubyte'.format(kind))

    with open(labels_path, 'rb') as lbpath:

        file_protocol, num_items = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    images_path = os.path.join(path, '{}-images-idx3-ubyte'.format(kind))

    with open(images_path, 'rb') as impath:

        magic, num, rows, cols = struct.unpack('>IIII', impath.read(16))
        images = np.fromfile(impath, dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255) - 0.5) * 2

        return images, labels


# Import the dataset

X_train, y_train = load_mnist()