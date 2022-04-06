""" MULTILAYER PERCEPTRON - GRADIENT DESCENT - PYTORCH
    ------------------------------------------------------
    Implementation of a multilayer perceptron for multi-class classification, with two hidden layers, using pytorch.
"""


# -------------------------------------------------------------------------------
# 0. IMPORT LIBRARIES
# -------------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import torch


# -------------------------------------------------------------------------------
# 1. PREPARE THE DATA
# -------------------------------------------------------------------------------


# Import the dataset

iris = load_iris()
X, y = iris['data'], iris['target']


# Separate the data into train and test subsets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=1)


# Apply the standardization to scale the features (using overall mean and variance)

X_train_std = (X_train - np.mean(X_train)) / np.std(X_train)
X_test_std = (X_test - np.mean(X_train)) / np.std(X_train)


# Create tensors of features and labels

X_train_std = torch.from_numpy(X_train_std).float()
y_train = torch.from_numpy(y_train)

X_test_std = torch.from_numpy(X_test_std).float()
y_test = torch.from_numpy(y_test)


# Combine tensors into a joint dataset

ds_train = torch.utils.data.TensorDataset(X_train_std, y_train)


# Create a dataset loader

torch.manual_seed(1)
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=2, shuffle=True)


# -------------------------------------------------------------------------------
# 2. DESIGN THE MODEL
# -------------------------------------------------------------------------------


# Design the multilayer perceptron

class Model(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size):

        super().__init__()
        self.layer_1 = torch.nn.Linear(in_features=input_size, out_features=hidden_size, bias=True)
        self.layer_2 = torch.nn.Linear(in_features=hidden_size, out_features=output_size, bias=True)

    def forward(self, x):

        x = self.layer_1(x)
        x = torch.nn.Sigmoid()(x)
        x = self.layer_2(x)
        x = torch.nn.Softmax(dim=1)(x)

        return x


# Instantiate the multilayer perceptron

model = Model(input_size=X_train_std.shape[1], hidden_size=16, output_size=3)


# Specify the loss function

loss_function = torch.nn.CrossEntropyLoss()


# Specify the optimizer

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)


# -------------------------------------------------------------------------------
# 3. TRAIN THE MODEL
# -------------------------------------------------------------------------------


# Train the multilayer perceptron

n_epochs = 100
loss_hist = [0] * n_epochs
accuracy_hist = [0] * n_epochs

for epoch in range(n_epochs):

    for X_batch, y_batch in dl_train:

        """ Generate predictions """
        pred = model(X_batch)

        """ Calculate loss """
        loss = loss_function(pred, y_batch)

        """ Compute gradients """
        loss.backward()

        """ Update parameters using gradients """
        optimizer.step()

        """ Reset gradients to zero """
        optimizer.zero_grad()

        loss_hist[epoch] += loss.item() * y_batch.size(0)
        accuracy_hist[epoch] += (torch.argmax(pred, dim=1) == y_batch).float().sum()

    loss_hist[epoch] /= len(dl_train.dataset)
    accuracy_hist[epoch] /= len(dl_train.dataset)


# Plot the number of misclassifications per epoch

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

ax[0].plot(loss_hist)
ax[0].set_title('Train Loss', size=15)
ax[0].set_xlabel('Epoch', size=15)
ax[0].tick_params(axis='both', which='major', labelsize=15)
ax[0].grid()

ax[1].plot(accuracy_hist)
ax[1].set_title('Train Accuracy', size=15)
ax[1].set_xlabel('Epoch', size=15)
ax[1].tick_params(axis='both', which='major', labelsize=15)
ax[1].grid()

plt.tight_layout()
fig.savefig('images/12_multilayer_perceptron_gd_pt/Training_loss_and_accuracy_per_epoch.png')


# -------------------------------------------------------------------------------
# 4. EVALUATE THE MODEL
# -------------------------------------------------------------------------------


# Evaluate the multilayer perceptron

pred_test = model(X_test_std)
accuracy = (torch.argmax(pred_test, dim=1) == y_test).float().mean()
print('Test set accuracy: {}'.format(accuracy))


# -------------------------------------------------------------------------------
# 5. GENERAL
# -------------------------------------------------------------------------------


# Show plots

plt.show()

