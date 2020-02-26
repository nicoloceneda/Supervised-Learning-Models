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
from sklearn import datasets, model_selection, preprocessing, linear_model, metrics


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


# -------------------------------------------------------------------------------
# 2. TRAIN THE PERCEPTRON
# -------------------------------------------------------------------------------


# Initialize a perceptron object

ppn = linear_model.Perceptron(max_iter=40, eta0=0.1, random_state=1)


# Learn from the data via the fit method

ppn.fit(X_train_std, y_train)


# -------------------------------------------------------------------------------
# 3. MAKE PREDICTIONS
# -------------------------------------------------------------------------------


# Predict the classes of the samples in the test set

y_predict = ppn.predict(X_test_std)


# Evaluate the performance of the model

print('Number of misclassifications: {}'.format(np.sum(y_test != y_predict)))
print('Prediction accuracy: {}'.format(metrics.accuracy_score(y_test, y_predict)))


# -------------------------------------------------------------------------------
# 4. PLOT THE DECISION BOUNDARY AND VERIFY THAT THE TRAINING SAMPLE IS CLASSIFIED CORRECTLY
# -------------------------------------------------------------------------------


# Function to plot the decision boundary

def plot_decision_regions(X, y, classifier, resolution=0.02, test_idx=None):

    """ Create a colormap object.

        Generate a matrix with two columns, where rows are all possible combinations of all numbers from min-1 to max+1 of the two series of
        features. The matrix with two columns is needed because the perceptron was trained on a matrix with such shape.

        Use the predict method of the ppn to predict the class corresponding to all the possible combinations of features generated in the
        above matrix. The predict method will use the weights learnt during the training phase: since the number of misclassifications conv-
        erged during the training phase, we expect the perceptron to find a decision boundary that correctly classifies all the samples in
        the training and test sets.

        Reshape the vector of predictions as the X0_grid.

        Draw filled contours, where all possible combinations of features are associated to a Z, which is 0, 1 or 2.

        To verify that the perceptron correctly classified all the samples in the training and test sets, plot the original features in the
        scatter plot and verify that they fall inside the correct region.

        Circle the sample belonging to the test set with a square.
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
        plt.scatter(X_test[:, 0], X_test[:, 1], alpha=0.8, linewidth=1, color='', marker='s', edgecolor='black', label='test_set')


# Plot the decision region and the data

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105, 150))
plt.title('Decision boundary and training sample')
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.savefig('images/03_perceptron_sl/Decision_boundary_and_training_sample.png')


# -------------------------------------------------------------------------------
# 5. GENERAL
# -------------------------------------------------------------------------------


# Show plots

plt.show()