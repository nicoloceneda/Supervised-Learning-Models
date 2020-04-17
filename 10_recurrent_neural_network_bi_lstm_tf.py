""" RECURRENT NEURAL NETWORK - BIDIRECTIONAL - LSTM - TENSOR FLOW
    -------------------------------------------------------------
    Implementation of a bidirectional lstm multilayer recurrent neural network for sentiment analysis, with a many-to-one architecture and
    two hidden layers, using tensorflow.
"""


# -------------------------------------------------------------------------------
# 0. IMPORT LIBRARIES
# -------------------------------------------------------------------------------


import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from collections import Counter


# -------------------------------------------------------------------------------
# 1. PREPARE THE DATA
# -------------------------------------------------------------------------------


# Import the dataset

imdb = pd.read_csv('imdb dataset/extracted/imdb_data.csv', encoding='utf-8')


# Create a tensorflow dataset with tuples of features and class labels

target = imdb.pop('sentiment')
ds = tf.data.Dataset.from_tensor_slices((imdb.values, target.values))


# Separate the data it into train, test and validation subsets

tf.random.set_seed(1)
ds_orig = ds.shuffle(50000, reshuffle_each_iteration=False)
ds_raw_test = ds_orig.take(25000)
ds_raw_train = ds_orig.skip(25000).take(20000)
ds_raw_valid = ds_orig.skip(25000).skip(20000)


# Identify the unique words (tokens) in the training dataset

text_into_words = tfds.features.text.Tokenizer()
token_and_counts = Counter()

for sample in ds_raw_train:

    tokens = text_into_words.tokenize(sample[0].numpy()[0])
    token_and_counts.update(tokens)


# Map each unique word to a unique integer

encoder = tfds.features.text.TokenTextEncoder(token_and_counts)


# Converted the sequences of words into sequences of integers

def encode(text_tensor, label):

    text = text_tensor.numpy()[0]
    encoded_text = encoder.encode(text)

    return encoded_text, label


def encode_map_fn(text, label):

    return tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))


ds_train = ds_raw_train.map(encode_map_fn)
ds_valid = ds_raw_valid.map(encode_map_fn)
ds_test = ds_raw_test.map(encode_map_fn)


# Divide the dataset into mini-batches, padding the shorter sequencies

batch_size = 32

ds_train = ds_train.padded_batch(batch_size, padded_shapes=([-1], []))
ds_valid = ds_valid.padded_batch(batch_size, padded_shapes=([-1], []))
ds_test = ds_test.padded_batch(batch_size, padded_shapes=([-1], []))


# -------------------------------------------------------------------------------
# 2. DESIGN THE MODEL
# -------------------------------------------------------------------------------


# Design the bidirectional lstm multilayer recurrent neural network

vocabulary_size = len(token_and_counts) + 2
embedding_size = 20
tf.random.set_seed(1)

bid_lstm_model = tf.keras.Sequential()
bid_lstm_model.add(tf.keras.layers.Embedding(input_dim=vocabulary_size, output_dim=embedding_size, name='embed-layer'))
bid_lstm_model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, name='lstm-layer'), name='bdir-lstm'))
bid_lstm_model.add(tf.keras.layers.Dense(64, activation='relu'))
bid_lstm_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


# Print the model summary

bid_lstm_model.summary()


# Compile the model to specify optimizer, loss function, evaluation metrics

bid_lstm_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=['accuracy'])


# -------------------------------------------------------------------------------
# 3. TRAIN THE MODEL
# -------------------------------------------------------------------------------


# Train the bidirectional lstm multilayer recurrent neural network

history = bid_lstm_model.fit(ds_train, validation_data=ds_valid, epochs=10)


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
plt.savefig('images/10_recurrent_neural_network_bi_lstm_tf/Training_loss_and_accuracy_per_epoch.png')


# -------------------------------------------------------------------------------
# 4. EVALUATE THE MODEL
# -------------------------------------------------------------------------------


# Evaluate the bidirectional lstm multilayer recurrent neural network

results = bid_lstm_model.evaluate(ds_test)
print('Test accuracy: {:.4f}'.format(results[1]*100))


# -------------------------------------------------------------------------------
# 5. GENERAL
# -------------------------------------------------------------------------------


# Show plots

plt.show()
