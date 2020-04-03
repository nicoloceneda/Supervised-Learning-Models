""" MULTILAYER PERCEPTRON - GRADIENT DESCENT - TENSOR FLOW
    ------------------------------------------------------
    Implementation of a multilayer perceptron for multi-class classification, with two hidden layer, using tensorflow.
"""


# -------------------------------------------------------------------------------
# 0. IMPORT LIBRARIES
# -------------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds


# -------------------------------------------------------------------------------
# 1. PREPARE THE DATA
# -------------------------------------------------------------------------------


# Import the dataset

iris, iris_info = tfds.load('iris', with_info=True)


# Print the dataset information

print(iris_info)


# Separate the data into train and test subsets

tf.random.set_seed(1)
ds_orig = iris['train'].shuffle(150, reshuffle_each_iteration=False)
ds_train = ds_orig.take(100)
ds_test = ds_orig.skip(100)


# Create tuples of features and class labels

ds_train = ds_train.map(lambda item: (item['features'], item['label']))
ds_test = ds_test.map(lambda item: (item['features'], item['label']))


# -------------------------------------------------------------------------------
# 2. DESIGN THE MODEL
# -------------------------------------------------------------------------------


# Design the multilayer perceptron

layers_1 = tf.keras.layers.Dense(units=16, activation='sigmoid', name='fc1', input_shape=(4, ))
layers_2 = tf.keras.layers.Dense(units=3, activation='softmax', name='fc2')
iris_model = tf.keras.Sequential([layers_1, layers_2])


# Print the model summary

iris_model.summary()


# Compile the model to specify optimizer, loss function, evaluation metrics

iris_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# -------------------------------------------------------------------------------
# 3. TRAIN THE MODEL
# -------------------------------------------------------------------------------


# Determine the number of steps for each epoch

num_epochs = 100
training_size = 100
batch_size = 2
steps_per_epoch = np.ceil(training_size / batch_size)


# Train the multilayer perceptron

ds_train = ds_train.shuffle(buffer_size=training_size)
ds_train = ds_train.repeat()
ds_train = ds_train.batch(batch_size=batch_size)
ds_train = ds_train.prefetch(buffer_size=1000)
history = iris_model.fit(ds_train, epochs=num_epochs, steps_per_epoch=steps_per_epoch, verbose=0)


# Visualize the learning curve

hist = history.history
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
ax[0].plot(hist['loss'])
ax[0].set_xlabel('Epoch')
ax[0].set_title('Training loss')
ax[0].tick_params(axis='both', which='major')
ax[1].plot(hist['accuracy'])
ax[1].set_xlabel('Epoch')
ax[1].set_title('Training accuracy')
ax[1].tick_params(axis='both', which='major')


# -------------------------------------------------------------------------------
# 4. EVALUATE THE MODEL
# -------------------------------------------------------------------------------


# Evaluate the multilayer perceptron

results = iris_model.evaluate(ds_test.batch(50), verbose=0)
print('Test loss: {:.4f} | Test accuracy: {:.4f}'.format(*results))


# -------------------------------------------------------------------------------
# 5. GENERAL
# -------------------------------------------------------------------------------


# Show plots

plt.show()
