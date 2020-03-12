""" multilayer_perceptron
    ---------------------
    Implementation of a multilayer perceptron.
"""


# ------------------------------------------------------------------------------------------------------------------------------------------
# 0. IMPORT LIBRARIES AND/OR MODULES
# ------------------------------------------------------------------------------------------------------------------------------------


import sys
import numpy as np
import matplotlib.pyplot as plt
import time


# Clock

start = time.time()


# ------------------------------------------------------------------------------------------------------------------------------------------
# 2. PREPARE THE DATA
# ------------------------------------------------------------------------------------------------------------------------------------------


# Load the pre-processed MNIST image arrays

mnist = np.load('mnist_scaled_c.npz')
X_train = mnist['X_train']
y_train = mnist['y_train']

# ------------------------------------------------------------------------------------------------------------------------------------------
# 1. DESIGN THE MULTILAYER PERCEPTRON
# ------------------------------------------------------------------------------------------------------------------------------------------


class NeuralNetMLP(object):

    """ Feedforward neural network / Multi-layer perceptron classifier.

    Parameters
    ----------
    n_hidden : int (default: 30)
        Number of hidden units.
    l2 : float (default: 0.)
        Lambda value for L2-regularization. No regularization if l2=0. (default)
    epochs : int (default: 100)
        Number of passes over the training set.
    eta : float (default: 0.001)
        Learning rate.
    shuffle : bool (default: True)
        Shuffles training data every epoch if True to prevent circles.
    minibatch_size : int (default: 1)
        Number of training samples per minibatch.
    seed : int (default: None)
        Random seed for initializing weights and shuffling.

    Attributes
    ----------
    eval : dict
      Dictionary collecting the cost, training accuracy and validation accuracy for each epoch during training.
    """

    def __init__(self, n_hidden=30, l2=0., epochs=100, eta=0.001, shuffle=True, minibatch_size=1, seed=None):

        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size
        self.random = np.random.RandomState(seed)

    def onehot(self, y, n_classes):

        """ Encode labels into one-hot representation

        Parameters
        ----------
        y : array, shape = [n_samples, ]
            Target values.
        n_classes : int
            Number of unique classes 

        Returns
        -------
        onehot : array, shape = (n_samples, n_labels)
        """

        onehot = np.zeros((y.shape[0], n_classes))

        for idx, val in enumerate(y.astype(int)):

            onehot[idx, val] = 1.

        return onehot

    def sigmoid(self, z):

        """ Compute logistic sigmoid function """

        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def forward(self, X):

        """ Compute forward propagation step """

        # Net input of hidden layer: [n_samples, n_features] o [n_features, n_hidden] -> [n_samples, n_hidden]
        z_h = np.dot(X, self.w_h) + self.b_h

        # Activation of hidden layer
        a_h = self.sigmoid(z_h)

        # Net input of output layer: [n_samples, n_hidden] o [n_hidden, n_classes] -> [n_samples, n_classes]
        z_out = np.dot(a_h, self.w_out) + self.b_out

        # Activation output layer
        a_out = self.sigmoid(z_out)

        return z_h, a_h, z_out, a_out

    def compute_cost(self, y_enc, output):
        
        """ Compute cost function.

        Parameters
        ----------
        y_enc : array, shape = (n_samples, n_labels)
            one-hot encoded class labels.
        output : array, shape = [n_samples, n_output_units]
            Activation of the output layer (forward propagation)

        Returns
        -------
        cost : float
            Regularized cost
        """
        
        l2_term = (self.l2 * (np.sum(self.w_h ** 2.) + np.sum(self.w_out ** 2.)))
        term1 = -y_enc * (np.log(output))
        term2 = (1. - y_enc) * np.log(1. - output)
        cost = np.sum(term1 - term2) + l2_term

        return cost

    def predict(self, X):

        """Predict class labels

        Parameters
        ----------
        X : array, shape = [n_samples, n_features]
            Input layer with original features.

        Returns
        -------
        y_pred : array, shape = [n_samples]
            Predicted class labels.
        """

        z_h, a_h, z_out, a_out = self.forward(X)
        y_pred = np.argmax(z_out, axis=1)
        return y_pred

    def fit(self, X_train, y_train, X_valid, y_valid):

        """ Learn weights from training data.

        Parameters
        ----------
        X_train : array, shape = [n_samples, n_features]
            Input layer with original features.
        y_train : array, shape = [n_samples, ]
            Target class labels.
        X_valid : array, shape = [n_samples, n_features]
            Sample features for validation during training
        y_valid : array, shape = [n_samples, ]
            Sample labels for validation during training

        Returns
        -------
        self
        """

        n_classes = len(np.unique(y_train))
        n_features = X_train.shape[1]

        # Initialization of weights and biases
        self.w_h = self.random.normal(loc=0.0, scale=0.1, size=(n_features, self.n_hidden))
        self.b_h = np.zeros(self.n_hidden)
        self.w_out = self.random.normal(loc=0.0, scale=0.1, size=(self.n_hidden, n_classes))
        self.b_out = np.zeros(n_classes)

        # One-hot representation of the vector of target classes
        y_train_onehot = self.onehot(y_train, n_classes)

        # Evaluation dictionary
        self.eval = {'cost': [], 'train_acc': [], 'valid_acc': []}

        # Iterate for all epochs
        for i in range(self.epochs):

            # Iterate for all minibatches
            indices = np.arange(X_train.shape[0])

            if self.shuffle:

                self.random.shuffle(indices)

            for start_idx in range(0, indices.shape[0] - self.minibatch_size + 1, self.minibatch_size):

                batch_idx = indices[start_idx:start_idx + self.minibatch_size]

                # FORWARD PROPAGATION

                z_h, a_h, z_out, a_out = self.forward(X_train[batch_idx])

                # BACK PROPAGATION

                # [n_samples, n_classes]
                sigma_out = a_out - y_train_onehot[batch_idx]

                # [n_samples, n_hidden]
                sigmoid_derivative_h = a_h * (1. - a_h)

                # ([n_samples, n_classes] dot [n_classes, n_hidden]) * [n_samples, n_hidden] -> [n_samples, n_hidden]
                sigma_h = (np.dot(sigma_out, self.w_out.T) * sigmoid_derivative_h)

                # Regularization and weight updates
                delta_w_h = np.dot(X_train[batch_idx].T, sigma_h) + self.l2 * self.w_h # DONE
                delta_b_h = np.sum(sigma_h, axis=0)
                delta_w_out = np.dot(a_h.T, sigma_out) + self.l2 * self.w_out  # DONE
                delta_b_out = np.sum(sigma_out, axis=0)

                self.w_h -= self.eta * delta_w_h # DONE
                self.b_h -= self.eta * delta_b_h
                self.w_out -= self.eta * delta_w_out # DONE
                self.b_out -= self.eta * delta_b_out

            # Evaluation after each epoch during training
            z_h, a_h, z_out, a_out = self.forward(X_train)

            cost = self.compute_cost(y_enc=y_train_onehot, output=a_out)

            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_valid)

            train_acc = ((np.sum(y_train == y_train_pred)).astype(np.float) / X_train.shape[0])
            valid_acc = ((np.sum(y_valid == y_valid_pred)).astype(np.float) / X_valid.shape[0])

            # Print the evolution of the training and validation accuracy per epoch
            
            sys.stderr.write('\r%0*d/%d | Cost: %.2f | Train/Valid Acc.: %.2f%%/%.2f%% ' % (len(str(self.epochs)), i + 1, self.epochs, cost, train_acc * 100, valid_acc * 100))
            sys.stderr.flush()

            self.eval['cost'].append(cost)
            self.eval['train_acc'].append(train_acc)
            self.eval['valid_acc'].append(valid_acc)

        return self


# ------------------------------------------------------------------------------------------------------------------------------------------
# 3. TRAIN THE MULTILAYER PERCEPTRON
# ------------------------------------------------------------------------------------------------------------------------------------------

# Initialize a 784-100-10 multilayer perceptron object

nn = NeuralNetMLP(n_hidden=100, l2=0.01, epochs=200, eta=0.0005, minibatch_size=100, shuffle=True, seed=1)


# Learn from the data

nn.fit(X_train=X_train[:55000], y_train=y_train[:55000], X_valid=X_train[55000:], y_valid=y_train[55000:])


# Plot the cost

plt.plot(range(nn.epochs), nn.eval['cost'])
plt.ylabel('Cost')
plt.xlabel('Epochs')
#plt.show()


# Plot the training and validation accuracy

plt.plot(range(nn.epochs), nn.eval['train_acc'], label='training')
plt.plot(range(nn.epochs), nn.eval['valid_acc'], label='validation', linestyle='--')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
#plt.show()


# Calculate the prediction accuracy

X_test = mnist['X_test']
y_test = mnist['y_test']

y_test_pred = nn.predict(X_test)
acc = (np.sum(y_test == y_test_pred).astype(np.float) / X_test.shape[0])

print('Test accuracy: %.2f%%' % (acc * 100))


# Visualize the images misclassified

miscl_img = X_test[y_test != y_test_pred][:25]
correct_lab = y_test[y_test != y_test_pred][:25]
miscl_lab = y_test_pred[y_test != y_test_pred][:25]

fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()

for i in range(25):

    img = miscl_img[i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[i].set_title('%d) t: %d p: %d' % (i+1, correct_lab[i], miscl_lab[i]))

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
#plt.show()


# Clock

end = time.time()

print(end - start)
