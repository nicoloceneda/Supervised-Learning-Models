""" SUPPORT VECTOR MACHINE - GRADIENT DESCENT - SCIKIT LEARN
    --------------------------------------------------------
    Implementation of a support vector machine for multi-class classification, via gradient descent algorithm, with standardized features,
    using scikit-learn.
"""


# -------------------------------------------------------------------------------
# 0. IMPORT LIBRARIES
# -------------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from sklearn import datasets, model_selection, preprocessing, svm, metrics


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
# 2. TRAIN THE SUPPORT VECTOR MACHINE
# -------------------------------------------------------------------------------


# Initialize a support vector machine object

svm = svm.SVC(C=1, random_state=1, kernel='linear')


# Learn from the data via the fit method

svm.fit(X_train_std, y_train)


# -------------------------------------------------------------------------------
# 3. MAKE PREDICTIONS
# -------------------------------------------------------------------------------


# Predict the classes of the samples in the test set

y_predict = svm.predict(X_test_std)


# Evaluate the performance of the model

print('Number of misclassifications: {}'.format(np.sum(y_test != y_predict)))
print('Prediction accuracy: {}'.format(metrics.accuracy_score(y_test, y_predict)))


# -------------------------------------------------------------------------------
# 4. PLOT THE DECISION BOUNDARY AND VERIFY THAT THE TRAINING AND TEST SAMPLES ARE CLASSIFIED CORRECTLY
# -------------------------------------------------------------------------------





