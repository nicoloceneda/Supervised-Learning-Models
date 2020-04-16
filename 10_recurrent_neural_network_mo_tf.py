""" RECURRENT NEURAL NETWORK - MANY TO ONE - TENSOR FLOW
    ----------------------------------------------------
    Implementation of a multilayer recurrent neural network for sentiment analysis, with a many-to-one architecture and two hidden layers,
    using tensorflow.
"""


# -------------------------------------------------------------------------------
# 0. IMPORT LIBRARIES
# -------------------------------------------------------------------------------


import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from collections import Counter


# -------------------------------------------------------------------------------
# 1. PREPARE THE DATA
# -------------------------------------------------------------------------------


# Import the dataset

imdb = pd.read_csv('imdb dataset/extracted/imdb_data.csv', encoding='utf-8')


# Create a tensorflow dataset

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

""" Note that there might be some tokens in the validation or testing data that are not present in the training data and are thus not included 
    in the mapping. If we have q tokens (that is the size of token_and_counts), then all tokens that haven't been seen before, and are thus 
    not included in token_and_counts, will be assigned the integer q + 1. In other words, the index q + 1 is reserved for unknown words. 
    Another reserved value is the integer 0, which serves as a placeholder for adjusting the sequence length. 
"""


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


# Create an embedding layer

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=100, output_dim=6, input_length=20, name='embed-layer'))
model.summary()


