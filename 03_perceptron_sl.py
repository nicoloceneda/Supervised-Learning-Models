""" PERCEPTRON_SL
    -------------
    Implementation of a single layer perceptron for multi-class classification via scikit-learn.
"""


# -------------------------------------------------------------------------------
# 0. IMPORT LIBRARIES
# -------------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from sklearn import datasets, model_selection, preprocessing, linear_model


# -------------------------------------------------------------------------------
# 1. PREPARE THE DATA
# -------------------------------------------------------------------------------


# Import the dataset

data = datasets.load_iris()


# Extract the class labels

y = data.target


# Extract the features

X = data.data[:, [2, 3]]


# Separate the data into train and test subsets with the same proportions of class labels as the input dataset

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)


# Apply the standardization to scale the features

std_scaler = preprocessing.StandardScaler()
std_scaler.fit(X_train)
X_train_std = std_scaler.transform(X_train)
X_test_std = std_scaler.transform(X_test)


# Plot the features in a scatter plot

plt.figure()
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='+', label='Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='+', label='Versicolor')
plt.scatter(X[100:, 0], X[100:, 1], color='lightgreen', marker='+', label='Virginica')
plt.title('Scatter plot of the features')
plt.xlabel('Petal length [cm]')
plt.ylabel('Petal width [cm]')
plt.legend(loc='upper left')
plt.savefig('images/03_perceptron_sl/Scatter_plot_of_the_features.png')


# -------------------------------------------------------------------------------
# 2. TRAIN THE PERCEPTRON
# -------------------------------------------------------------------------------


# Initialize a perceptron object

ppn = linear_model.Perceptron(max_iter=40, eta0=0.1, random_state=1)


# Learn from the data via the fit method

ppn.fit(X_train_std, y_train)


# -------------------------------------------------------------------------------
# 5. MAKE PREDICTIONS
# -------------------------------------------------------------------------------


y_pred = ppn.predict(X_test_std)
print('Number of misclassification: {}'.format(np.sum(y_test != y_pred)))


# -------------------------------------------------------------------------------
# 5. GENERAL
# -------------------------------------------------------------------------------


# Show plots

plt.show()