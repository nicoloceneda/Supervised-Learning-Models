""" perceptron
    ----------
    Implementation of a single layer perceptron via sci-kit learn.
"""


# ------------------------------------------------------------------------------------------------------------------------------------------
# 0. IMPORT LIBRARIES AND/OR MODULES
# ------------------------------------------------------------------------------------------------------------------------------------------


import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import matplotlib.colors as clr


# ------------------------------------------------------------------------------------------------------------------------------------------
# 1. PREPARE THE DATA
# ------------------------------------------------------------------------------------------------------------------------------------------


# Import the dataset

iris = load_iris()
print(iris)


# Extract the class labels

y = iris.target


# Extract the features

X = iris.data[:, [2, 3]]


# Plot the features in a scatter plot

plt.figure()
plt.scatter(X[:50, 0], X[:50, 1], color="red", edgecolor='black', marker="+", label="Setosa")
plt.scatter(X[50:100, 0], X[50:100, 1], color="blue", edgecolor='black', marker="+", label="Versicolor")
plt.scatter(X[100:150, 0], X[100:150, 1], color="lightgreen", edgecolor='black', marker="+", label="Virginica")
plt.title("Scatter plot of the features")
plt.xlabel("Petal length [cm]")
plt.ylabel("Petal width [cm]")
plt.legend(loc="upper left")
plt.savefig('images/03_perceptron/Scatter_plot_of_the_features.png')


# Separate the data into a train and a test subset with the same proportions of class labels as the input dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)


# Apply the standardization to scale the features (note that the same scaling parameters are used to standardize the test set)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# ------------------------------------------------------------------------------------------------------------------------------------------
# 2. TRAIN THE PERCEPTRON
# ------------------------------------------------------------------------------------------------------------------------------------------


# Initialize a perceptron object

ppn = Perceptron(n_iter=40, eta0=0.1, random_state=1)


# Learn from data via the fit method

ppn.fit(X_train_std, y_train)


# ------------------------------------------------------------------------------------------------------------------------------------------
# 3. MAKE PREDICTIONS
# ------------------------------------------------------------------------------------------------------------------------------------------


# Predict the classes of the features in the test set

y_pred = ppn.predict(X_test_std)


# Calculate the number of misclassifications

n_miscl = (y_test != y_pred).sum()
print('Misclassified samples: {}'.format(n_miscl))


# Calculate the classification accuracy

print('Accuracy: {}'.format(accuracy_score(y_test, y_pred)))
print('Accuracy: {}'.format(ppn.score(X_test_std, y_test)))


# ------------------------------------------------------------------------------------------------------------------------------------------
# 4. VISUALIZE THE DECISION BOUNDARIES AND VERIFY HOW WELL THE PERCEPTRON CLASSIFIES THE DIFFERENT SAMPLES
# ------------------------------------------------------------------------------------------------------------------------------------------


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    """ Create a colormap.

        Generate a matrix with two columns, where rows are all possible combinations of all numbers from min-1 to max+1 of the two series of
        features. The matrix with two columns is needed because the perceptron was trained on a matrix with such shape.

        Use the predict method of the chosen classifier (ppn) to predict the class corresponding to all the possible combinations of features
        generated in the above matrix. The predict method will use the weights learnt during the training phase.

        Reshape the vector of predictions as the X0_grid.

        Draw filled contours, where all possible combinations of features are associated to a Z, which is +1 or -1.

        To verify whether the perceptron correctly classified all possible combinations of the features, plot the the original features in the
        scatter plot and verify that they fall inside the correct region.
    """

    colors = ('red', 'blue', 'green')
    cmap = clr.ListedColormap(colors[:len(np.unique(y))])

    X0_min, X0_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    X1_min, X1_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    X0_grid, X1_grid = np.meshgrid(np.arange(X0_min, X0_max, resolution), np.arange(X1_min, X1_max, resolution))
    X0X1_combs = np.array([X0_grid.ravel(), X1_grid.ravel()]).T

    Z = classifier.predict(X0X1_combs)

    Z = Z.reshape(X0_grid.shape)

    plt.figure()
    plt.contourf(X0_grid, X1_grid, Z, alpha=0.3, cmap=cmap)
    plt.xlim(X0_min, X0_max)
    plt.ylim(X1_min, X1_max)

    for pos, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, color=colors[pos], marker='+', label=cl)

    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], alpha=1.0, linewidth=1, color='', marker='s', edgecolor='black', label='test_set')


# Plot the decision region and the data

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105, 150))
plt.title('Decision boundary and training sample')
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.savefig('images/03_perceptron/Decision_boundary_and_training_sample.png')


# ------------------------------------------------------------------------------------------------------------------------------------------
# 5. GENERAL
# ------------------------------------------------------------------------------------------------------------------------------------------


# Show plots

plt.show()