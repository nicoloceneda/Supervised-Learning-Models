""" MULTILAYER PERCEPTRON - GRADIENT DESCENT
    ----------------------------------------
    Implementation of a multilayer perceptron for multi-class classification, with one hidden layer.
"""


# -------------------------------------------------------------------------------
# 0. IMPORT LIBRARIES
# -------------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
import sys


# -------------------------------------------------------------------------------
# 1. PREPARE THE DATA
# -------------------------------------------------------------------------------


# Import the dataset

mnist = np.load('mnist dataset/compressed/mnist_std.npz')

# Extract the class labels

y_train = mnist['y_train']
y_test = mnist['y_test']


# Extract the features

X_train_std = mnist['X_train_std']
X_test_std = mnist['X_test_std']


# -------------------------------------------------------------------------------
# 2. DESIGN THE MULTILAYER PERCEPTRON
# -------------------------------------------------------------------------------


""" Clarifying notation:

    n_samples = bs
    n_features = m (basis activation excluded)
    n_hidden = d (basis activation excluded)
    n_labels = t
"""

class MultilayerPerceptron:

    """ Multilayer Perceptron classifier

        Parameters:
        ----------
        eta : float
            Learning rate (between 0.0 and 1.0)
        n_epochs : int
            Number of epochs.
        shuffle : bool
            If set to true it shuffles the training set before each epoch to prevent cycles.
        l2 : float
            Lambda parameter for L2-regularization to decrease degree of overfitting.
        n_samples_mb : int
            Number of training samples per mini-batch.
        n_hidden : int
            Number of units in the hidden layer.


        Attributes:
        ----------
        eval_train : dict
            Dictionary containing the cost, training accuracy and validation accuracy for each epoch during training.
    """

    def __init__(self, eta=0.01, n_epochs=100, shuffle=True, l2=0.0, n_samples_mb=1, n_hidden=30):

        self.eta = eta
        self.n_epochs = n_epochs
        self.shuffle = shuffle
        self.l2 = l2
        self.n_samples_mb = n_samples_mb
        self.n_hidden = n_hidden

    def one_hot_encode(self, y_train, n_labels):

        """ Encode the labels into the one-hot representation
            (Used in fit method - weight initialization step)

            Parameters:
            ----------
            y_train : array, shape = [n_samples in train, ]
            n_labels : int

            Returns:
            -------
            onehot : array, shape = [n_samples in train, n_labels]
        """

        onehot = np.zeros((y_train.shape[0], n_labels))

        for sample, label in enumerate(y_train.astype(int)):

            onehot[sample, label] = 1

        return onehot

    def sigmoid_activ(self, Z):

        """ Return the probability level after the logistic sigmoid function
            (Used in forward_propagate method)

            Parameters:
            ----------
            Z : array, shape = see forward_propagate method

            Returns:
            -------
            sigmoid_active : same as Z
        """

        return 1 / (1 + np.exp(-np.clip(Z, -250, 250)))

    def forward_propagate(self, A_in):

        """ Compute the forward propagation step
            (Used in predict method and fit method {forward propagation, evaluation})

            Parameters:
            ----------
            A_in : array, shape = [n_samples in train/valid/test, n_features] in predict method
                   array, shape = [n_samples_mb, n_features] in fit method {forward propagation}
                   array, shape = [n_samples in train, n_features] in fit method {evaluation}

            Returns:
            -------
            In predict method:
            Z_h : array, shape = [n_hidden, ] + [n_samples in tr/va/te, n_features] * [n_features, n_hidden] = [n_samples in tr/va/te, n_hidden]
            A_h : array, shape = [n_samples in tr/va/te, n_hidden]
            Z_out : array, shape = [n_labels, ] + [n_samples in tr/va/te, n_hidden] * [n_hidden, n_labels] = [n_samples in tr/va/te, n_labels]
            A_out : array, shape = [n_samples in tr/va/te, n_labels]

            In fit method {forward propagation}:
            Z_h : array, shape = [n_hidden, ] + [n_samples_mb, n_features] * [n_features, n_hidden] = [n_samples_mb, n_hidden]
            A_h : array, shape = [n_samples_mb, n_hidden]
            Z_out : array, shape = [n_labels, ] + [n_samples_mb, n_hidden] * [n_hidden, n_labels] = [n_samples_mb, n_labels]
            A_out : array, shape = [n_samples_mb, n_labels]

            In fit method {evaluation}:
            Z_h : array, shape = [n_hidden, ] + [n_samples in train, n_features] * [n_features, n_hidden] = [n_samples in train, n_hidden]
            A_h : array, shape = [n_samples in train, n_hidden]
            Z_out : array, shape = [n_labels, ] + [n_samples in train, n_hidden] * [n_hidden, n_labels] = [n_samples in train, n_labels]
            A_out : array, shape = [n_samples in train, n_labels]
        """

        Z_h = self.b_h + np.dot(A_in, self.W_h)
        A_h = self.sigmoid_activ(Z_h)

        Z_out = self.b_out + np.dot(A_h, self.W_out)
        A_out = self.sigmoid_activ(Z_out)

        return Z_h, A_h, Z_out, A_out

    def predict(self, X):

        """ Predict class labels
            (Used in fit method {evaluation} and test step)

            Parameters:
            ----------
            X : array, shape = [n_samples in train, n_features] in fit method {evaluation}
                array, shape = [n_samples in valid, n_features] in fit method {evaluation}
                array, shape = [n_samples in test, n_features] in test section

            Returns:
            -------
            y_pred : array, shape = [n_samples in train, n_labels] in fit method {evaluation}
                     array, shape = [n_samples in valid, n_labels] in fit method {evaluation}
                     array, shape = [n_samples in test, n_labels] in test section
        """

        Z_h, A_h, Z_out, A_out = self.forward_propagate(X)
        y_pred = np.argmax(Z_out, axis=1)

        return y_pred

    def cost_function(self, y_train_one_hot, A_out):

        """ Compute cost function
            (Used in fit method {evaluation})

            Parameters:
            ----------
            y_train_one_hot : array, shape = [n_samples in train, n_labels]
            A_out : array, shape = [n_samples in train, n_labels]

            Returns:
            -------
            cost_function : float
        """

        l2_term = self.l2 * (np.sum(self.W_h ** 2) + np.sum(self.W_out ** 2))

        return np.sum(- y_train_one_hot * np.log(A_out) - (1 - y_train_one_hot) * np.log(1 - A_out)) + l2_term

    def fit(self, X_train_std, y_train, X_valid_std, y_valid):

        """ Learn the weights from the training data

            Parameters:
            ----------
            X_train_std : array, shape = [n_samples_train, n_features]
            y_train : array, shape = [n_samples_train, ]
            X_valid_std : array, shape = [n_samples_valid, n_features]
            y_valid : array, shape = [n_samples_valid, ]

            Returns:
            -------
            self : object
        """

        # Dataset characteristics

        n_features = X_train_std.shape[1]
        n_labels = np.unique(y_train).shape[0]
        n_samples_train = X_train_std.shape[0]

        # Initialize weights

        rgen = np.random.RandomState(seed=1)

        self.b_h = np.zeros(self.n_hidden)
        self.W_h = rgen.normal(loc=0.0, scale=0.1, size=(n_features, self.n_hidden))
        self.b_out = np.zeros(n_labels)
        self.W_out = rgen.normal(loc=0.0, scale=0.1, size=(self.n_hidden, n_labels))

        # Iterate over each epoch

        self.evaluation = {'cost': [], 'train_acc': [], 'valid_acc': []}
        y_train_enc = self.one_hot_encode(y_train, n_labels)

        for epoch in range(self.n_epochs):

            # Iterate over each minibatch

            indices = np.arange(n_samples_train)

            if self.shuffle:

                rgen.shuffle(indices)

            for start_mb in range(0, n_samples_train - self.n_samples_mb + 1, self.n_samples_mb):

                index_mb = indices[start_mb:start_mb + self.n_samples_mb]

                # Forward propagate

                z_h, A_h, z_out, A_out = self.forward_propagate(X_train_std[index_mb])

                # Back propagate

                delta_out = A_out - y_train_enc[index_mb]
                delta_h = np.dot(delta_out, self.W_out.T) * (A_h * (1 - A_h))

                grad_W_out = np.dot(A_h.T, delta_out)
                grad_b_out = np.sum(delta_out, axis=0)

                grad_W_h = np.dot(X_train_std[index_mb].T, delta_h)
                grad_b_h = np.sum(delta_h, axis=0)

                # Regularization and weight updates

                self.W_out -= self.eta * (grad_W_out + self.l2 * self.W_out)
                self.b_out -= self.eta * grad_b_out

                self.W_h -= self.eta * (grad_W_h + self.l2 * self.W_h)
                self.b_h -= self.eta * grad_b_h

            # Evaluation

            z_h, A_h, z_out, A_out = self.forward_propagate(X_train_std)

            cost = self.cost_function(y_train_enc, A_out)

            y_train_pred = self.predict(X_train_std)
            y_valid_pred = self.predict(X_valid_std)

            train_acc = (np.sum(y_train == y_train_pred)).astype(np.float) / X_train_std.shape[0]
            valid_acc = (np.sum(y_valid == y_valid_pred)).astype(np.float) / X_valid_std.shape[0]

            sys.stderr.write('\r%0*d/%d | Cost: %.2f | Train/Valid Acc.: %.2f%%/%.2f%% ' % (len(str(self.n_epochs)), epoch+1, self.n_epochs, cost, train_acc*100, valid_acc*100))
            sys.stderr.flush()

            self.evaluation['cost'].append(cost)
            self.evaluation['train_acc'].append(train_acc)
            self.evaluation['valid_acc'].append(valid_acc)

        return self


# -------------------------------------------------------------------------------
# 3. TRAIN THE MULTILAYER PERCEPTRON
# -------------------------------------------------------------------------------


# Initiate a multilayer perceptron object: n_features=784, n_hidden=100, n_output=10

mlp = MultilayerPerceptron(eta=0.0005, n_epochs=200, shuffle=True, l2=0.01, n_samples_mb=100, n_hidden=100)


# Learn from the data via the fit method: 55000 samples for training, 5000 samples for validation

mlp.fit(X_train_std[:55000], y_train[:55000], X_train_std[55000:], y_train[55000:])


# -------------------------------------------------------------------------------
# 4. MODEL EVALUATION
# -------------------------------------------------------------------------------


# Plot the cost function per epoch

plt.figure()
plt.plot(range(1, len(mlp.evaluation['cost']) + 1), mlp.evaluation['cost'])
plt.title('Cost function per epoch')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.savefig('images/08_multilayer_perceptron_gd/Number_of_misclassifications_per_epoch.png')


# Plot the training and validation accuracy per epoch

plt.figure()
plt.plot(range(1, len(mlp.evaluation['train_acc']) + 1), mlp.evaluation['train_acc'], label='train')
plt.plot(range(1, len(mlp.evaluation['valid_acc']) + 1), mlp.evaluation['valid_acc'], label='valid')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.savefig('images/08_multilayer_perceptron_gd/Train_and_valid_accuracy_per_epoch')


# Calculate the prediction accuracy (generalization performance)

y_test_predict = mlp.predict(X_test_std)
accuracy = np.sum(y_test == y_test_predict).astype(np.float) / X_test_std.shape[0]
print('Training accuracy: {}%'.format(accuracy * 100))


# Plot some of the images that have been misclassified

misclassified_images = X_test_std[y_test != y_test_predict][:25]
misclassified_label = y_test_predict[y_test != y_test_predict][:25]
correct_label = y_test[y_test != y_test_predict][:25]

fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()

for i in range(25):

    img = misclassified_images[i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[i].set_title('{}) t: {} p: {}'.format(i + 1, misclassified_label[i], correct_label[i]))

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.savefig('images/08_multilayer_perceptron_gd/Examples_of_misclassified_images.png')


# -------------------------------------------------------------------------------
# 5. GENERAL
# -------------------------------------------------------------------------------


# Show plots

plt.show()
