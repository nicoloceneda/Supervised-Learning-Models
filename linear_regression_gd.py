""" LINEAR REGRESSION - GRADIENT DESCENT
    ------------------------------------
    Implementation of an ordinary least squares regression for regression analysis, via gradient descent algorithm, with standardized
    features.
"""


# -------------------------------------------------------------------------------
# 0. IMPORT LIBRARIES
# -------------------------------------------------------------------------------


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# -------------------------------------------------------------------------------
# 1. PREPARE THE DATA
# -------------------------------------------------------------------------------


# Import the dataset

columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area', 'Central Air', 'Total Bsmt SF', 'SalePrice']
data = pd.read_csv('http://jse.amstat.org/v19n3/decock/AmesHousing.txt', sep='\t', usecols=columns)
print(data.head())


# Convert feature from string to binary

data['Central Air'] = data['Central Air'].map({'N': 0, 'Y': 1})


# Check for missing data and remove observations

print(data.isna().sum())
data = data.dropna(axis=0)


# Extract the class labels

y = data['SalePrice'].values


# Extract the features

X = data[['Gr Liv Area']].values


# Apply the standardization to scale the features and target variable

X_std = StandardScaler().fit_transform(X)
y_std = StandardScaler().fit_transform(y[:, np.newaxis]).flatten()


# -------------------------------------------------------------------------------
# 2. DESIGN THE MODEL
# -------------------------------------------------------------------------------


# Design the adaline

class LinearRegressionGD:

    """ Linear regression

        Parameters:
        ----------
        eta : float
            Learning rate (between 0.0 and 1.0).
        n_epochs : int
            Number of epochs.

        Attributes:
        ----------
        w : array, shape = [n_features, ]
            Weights after fitting.
        b : scalar
            Bias unit after fitting.
        loss_fun : list
            Mean squared error loss function in each epoch.
    """

    def __init__(self, eta=0.01, n_epochs=100):

        self.eta = eta
        self.n_epochs = n_epochs

    def fit(self, X, y):

        """ Fit training set

            Parameters:
            ----------
            X : array, shape = [n_samples, n_features]
                Training vectors.
            y : array, shape = [n_samples, ]
                Target values.

            Returns:
            -------
            self : object
        """

        rgen = np.random.RandomState(seed=1)
        self.w = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b = 0.0
        self.loss_fun = []

        for epoch in range(self.n_epochs):

            update = y - self.linear_activ(X)
            self.w += self.eta * np.dot(X.T, update) / X.shape[0]
            self.b += self.eta * np.sum(update) / X.shape[0]
            loss = 0.5 * np.mean(update ** 2)
            self.loss_fun.append(loss)

        return self

    def linear_activ(self, X):

        """ Calculate the net input and return the linear activation
            (Used in the fit method)

            Parameters:
            ----------
            X : array, shape = [n_samples, n_features]

            Returns:
            -------
            net_input : array, shape = [n_samples, ]
        """

        net_input = np.dot(X, self.w) + self.b

        return net_input


# -------------------------------------------------------------------------------
# 3. TRAIN THE MODEL
# -------------------------------------------------------------------------------


# Initialize an adaline object

lr = LinearRegressionGD(eta=0.1, n_epochs=50)


# Learn from the data via the fit method

lr.fit(X_std, y_std)


# Plot the loss function per epoch

plt.figure()
plt.plot(range(1, lr.n_epochs + 1), lr.loss_fun, marker='o')
plt.title('Loss function per epoch')
plt.xlabel("Epoch")
plt.ylabel('Mean squared errors')
plt.savefig('images/09_linear_regression_gd/Loss_function_per_epoch.png')


# -------------------------------------------------------------------------------
# 4. EVALUATE THE MODEL
# -------------------------------------------------------------------------------


# Plot the best fit line

plt.figure()
plt.scatter(X_std, y_std, color="blue", marker="o", edgecolor='black')
plt.plot(X_std, lr.linear_activ(X_std), color='black', lw=2)
plt.title("Scatter plot of the dependent and independent variable")
plt.xlabel("Gr Liv Area")
plt.ylabel("SalePrice")
plt.savefig('images/09_linear_regression_gd/Best_fit_line.png')


# -------------------------------------------------------------------------------
# 5. GENERAL
# -------------------------------------------------------------------------------


# Show plots

plt.show()

