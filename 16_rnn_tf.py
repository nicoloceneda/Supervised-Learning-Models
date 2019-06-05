""" rnn_tf
    -----
    Implementation of a recurrent neural network via Tensorflow.
"""


# ------------------------------------------------------------------------------------------------------------------------------------------
# 0. IMPORT LIBRARIES AND/OR MODULES
# ------------------------------------------------------------------------------------------------------------------------------------------


import pandas as pd
from collections import Counter
import pyprind
import tensorflow as tf
from string import punctuation
import re
import numpy as np


# ------------------------------------------------------------------------------------------------------------------------------------------
# 1. LOAD AND PREPARE THE DATA
# ------------------------------------------------------------------------------------------------------------------------------------------


# Load the data from csv file

df = pd.read_csv('movie_data.csv', encoding='utf-8')


# Separate words (including punctuation) in each review and count each word's occurrence in counts

counts = Counter()

pbar = pyprind.ProgBar(len(df['review']), title='Counting words occurrence')

for pos, review in enumerate(df['review']):

    text = ''.join([c if c not in punctuation else ' '+c+' ' for c in review]).lower()
    df.loc[pos, 'review'] = text
    counts.update(text.split())
    pbar.update()


# Create a mapping of each unique word to an integer where the most frequent word corresponds to 1

words_counts = sorted(counts, key=counts.get, reverse=True)
word_to_int = {word: pos for pos, word in enumerate(words_counts, 1)}


# Rewrite the sentences as numbers in mapped_reviews

mapped_reviews = []

pbar = pyprind.ProgBar(len(df['review']), title='Map reviews to ints')

for review in df['review']:

    mapped_reviews.append([word_to_int[word] for word in review.split()])
    pbar.update()


# Create a matrix of same-length sequences: if a sequence < 200 then left-pad it with zeros, if a sequence > 200 then use the last 200 elements

sequence_length = 200
sequences = np.zeros((len(mapped_reviews), sequence_length), dtype=int)

for pos, row in enumerate(mapped_reviews):

    review_arr = np.array(row)
    sequences[pos, -len(row):] = review_arr[- sequence_length:]


# Separate the dataset for training and testing (the dataset has already been shuffled)

X_train = sequences[:25000, :]
y_train = df.loc[:25000, 'sentiment'].to_numpy()
X_test = sequences[25000:, :]
y_test = df.loc[25000:, 'sentiment'].to_numpy()


# Create a function that breaks a given dataset into chunks and returns a generator to iterate through this chunk

np.random.seed(123)

def create_batch_generator(x, y=None, batch_size=64):

    n_batches = len(x) // batch_size
    x = x[:n_batches * batch_size]

    if y is not None:

        y = y[:n_batches * batch_size]

    for pos in range(0, len(x), batch_size):

        if y is not None:

            yield x[pos:pos + batch_size], y[pos:pos + batch_size]

        else:

            yield x[pos:pos + batch_size]


# ------------------------------------------------------------------------------------------------------------------------------------------
# 2. CONSTRUCT THE RECURRENT NEURAL NETWORK
# ------------------------------------------------------------------------------------------------------------------------------------------


class SentimentRNN(object):

    def __init__(self, n_words, embed_size=200, seq_len=200, lstm_size=256, num_layers=1, batch_size=64, learning_rate=0.0001):

        self.n_words = n_words
        self.embed_size = embed_size
        self.seq_len = seq_len
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Create the computational graph and build the RNN

        self.g = tf.Graph()

        with self.g.as_default():

            tf.set_random_seed(123)
            self.build()
            self.saver = tf.train.Saver()
            self.init_op = tf.global_variables_initializer()

    def build(self):

        # Placeholders

        tf_x = tf.placeholder(dtype=tf.int32, shape=(self.batch_size, self.seq_len), name='tf_x')
        tf_y = tf.placeholder(dtype=tf.float32, shape=self.batch_size, name='tf_y')
        tf_keepprob = tf.placeholder(dtype=tf.float32, name='tf_keepprob')  # TODO: for the dropout configuration of the hidden layer

        # Embedding layer

        embedding = tf.Variable(tf.random_uniform((self.n_words, self.embed_size), minval=-1, maxval=1), name='embedding')
        embed_x = tf.nn.embedding_lookup(embedding, tf_x, name='embeded_x')  # TODO: matrix (batch_size, embed_size) where each row of numbers become a row of line numbers

        # Define the cells with BasicLSTMCell, apply the dropout to them and stack them in a list to form a multilayer RNN with MultiRNNCell

        cells_one_layer = tf.contrib.rnn.BasicLSTMCell(self.lstm_size)
        cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(cells_one_layer, output_keep_prob=tf_keepprob) for i in range(self.num_layers)])

        # Define the initial state of the cells

        self.initial_state = cells.zero_state(self.batch_size, tf.float32)  # TODO: [batch_size, state_size] filled with zeros
        print('\nInitial state:\n', self.initial_state)

        # Pull the embedded data, the RNN cells and their initial states and create a pipeline according to the unrolled LSTM architecture

        lstm_outputs, self.final_state = tf.nn.dynamic_rnn(cells, embed_x, initial_state=self.initial_state)  # TODO: output is matrix (batch_size, num_steps, lstm_size)
        print('\nLSTM_output:\n', lstm_outputs)
        print('\nFinal state:\n', self.final_state)

        # Pass outputs to a fully connected layer to get logits

        logits = tf.layers.dense(inputs=lstm_outputs[:, -1], units=1, activation=None, name='logits')
        logits = tf.squeeze(logits, name='logits_squeezed')
        print('\nLogits:\n', logits)

        # Predict

        y_proba = tf.nn.sigmoid(logits, name='probabilities')
        print('\nPredictions:\n', {'probabilities': y_proba, 'labels': tf.cast(tf.round(y_proba), tf.int32, name='labels')})

        # Define the cost function

        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_y, logits=logits), name='cost')

        # Define the optimizer

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.minimize(cost, name='train_op')

    def train(self, X_train, y_train, num_epochs):

        with tf.Session(graph=self.g) as sess:

            sess.run(self.init_op)
            iteration = 1

            for epoch in range(num_epochs):

                state = sess.run(self.initial_state)

                for batch_x, batch_y in create_batch_generator(X_train, y_train, self.batch_size):

                    feed = {'tf_x:0': batch_x, 'tf_y:0': batch_y, 'tf_keepprob:0': 0.5, self.initial_state: state}
                    loss, _, state = sess.run(['cost:0', 'train_op', self.final_state], feed_dict=feed)

                    if iteration % 20 == 0:

                        print("Epoch: {}/{} Iteration: {} | Train loss: {:.5f}".format(epoch + 1, num_epochs, iteration, loss))

                    iteration += 1

                if (epoch + 1) % 10 == 0:

                    self.saver.save(sess, "model/sentiment-{}.ckpt".format(epoch))

    def predict(self, X_data, return_proba=False):

        preds = []

        with tf.Session(graph=self.g) as sess:

            self.saver.restore(sess, tf.train.latest_checkpoint('model/'))

            test_state = sess.run(self.initial_state)

            for ii, batch_x in enumerate(create_batch_generator(X_data, None, batch_size=self.batch_size), 1):

                feed = {'tf_x:0': batch_x, 'tf_keepprob:0': 1.0, self.initial_state: test_state}

                if return_proba:

                    pred, test_state = sess.run(['probabilities:0', self.final_state], feed_dict=feed)

                else:

                    pred, test_state = sess.run(['labels:0', self.final_state], feed_dict=feed)

                preds.append(pred)

        return np.concatenate(preds)


# ------------------------------------------------------------------------------------------------------------------------------------------
# 3. TRAIN AND FORECAST
# ------------------------------------------------------------------------------------------------------------------------------------------


# Instantiate a RNN

n_words = max(list(word_to_int.values())) + 1
rnn = SentimentRNN(n_words=n_words, seq_len=sequence_length, embed_size=256, lstm_size=128, num_layers=1, batch_size=100, learning_rate=0.001)


# Train the RNN

rnn.train(X_train, y_train, num_epochs=40)


# Make predictions

preds = rnn.predict(X_test)
y_true = y_test[:len(preds)]
print('Test Acc.: {:.3f}'.format(np.sum(preds == y_true) / len(y_true)))
proba = rnn.predict(X_test, return_proba=True)