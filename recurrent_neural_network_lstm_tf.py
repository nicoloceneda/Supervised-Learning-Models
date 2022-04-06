""" RECURRENT NEURAL NETWORK - LSTM - TENSOR FLOW
    ---------------------------------------------
    Implementation of a lstm multilayer recurrent neural network for text generation, with a many-to-many architecture and two hidden layers,
    using tensorflow.
"""


# -------------------------------------------------------------------------------
# 0. IMPORT LIBRARIES
# -------------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# -------------------------------------------------------------------------------
# 1. PREPARE THE DATA
# -------------------------------------------------------------------------------


# Import the dataset

with open('gutenberg/gutenberg.txt', 'r') as fp:

    text = fp.read()

start_index = text.find('THE MYSTERIOUS ISLAND')
end_index = text.find('End of the Project Gutenberg')
text = text[start_index:end_index]


# Map each unique word to a unique integer

character_set = sorted(set(text))
character_to_integer = {ch: i for i, ch in enumerate(character_set)}


# Convert the text into numbers

text_encoded = np.array([character_to_integer[ch] for ch in text], dtype=np.int32)
print('Example of encoded text: ', text[15:21], '-->', text_encoded[15:21])


# Map each unique integer to a unique word

character_array = np.array(character_set)
print('Example of decoded text: ', text_encoded[15:21], '-->', ''.join(character_array[text_encoded[15:21]]))


# Create a tensorflow dataset

ds_text_encoded = tf.data.Dataset.from_tensor_slices(text_encoded)


# Divide the dataset into chunks

seq_length = 40
chunk_size = seq_length + 1
ds_chunks = ds_text_encoded.batch(chunk_size, drop_remainder=True)


# Create a function to split input and target sequences

def split_input_target(chunk):

    input_seq = chunk[:-1]
    target_seq = chunk[1:]

    return input_seq, target_seq


ds_sequences = ds_chunks.map(split_input_target)


# Divide the dataset into mini-batches

batch_size = 64
buffer_size = 10000
ds = ds_sequences.shuffle(buffer_size).batch(batch_size)


# -------------------------------------------------------------------------------
# 2. DESIGN THE MODEL
# -------------------------------------------------------------------------------


# Create a function to design the lstm multilayer recurrent neural network

def build_model(vocab_size, embedding_size, rnn_units):

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, embedding_size))
    model.add(tf.keras.layers.LSTM(rnn_units, return_sequences=True))
    model.add(tf.keras.layers.Dense(vocab_size))

    return model


# Design the lstm multilayer recurrent neural network

tf.random.set_seed(1)
lstm_model = build_model(vocab_size=len(character_set), embedding_size=256, rnn_units=512)


# Print the model summary

lstm_model.summary()


# Compile the model to specify optimizer, loss function, evaluation metrics

lstm_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])


# -------------------------------------------------------------------------------
# 3. TRAIN THE MODEL
# -------------------------------------------------------------------------------


# Train the lstm multilayer recurrent neural network

history = lstm_model.fit(ds, epochs=20)


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
plt.savefig('images/10_recurrent_neural_network_lstm_tf/Training_loss_and_accuracy_per_epoch.png')


# -------------------------------------------------------------------------------
# 4. EVALUATE THE MODEL
# -------------------------------------------------------------------------------

# Evaluate the multilayer recurrent neural network

def sample(model, starting_string, len_generated_text=500, max_input_length=40, scale_factor=1.0):

    encoded_input = [character_to_integer[s] for s in starting_string]
    encoded_input = tf.reshape(encoded_input, (1, -1))
    generated_string = starting_string

    model.reset_states()

    for i in range(len_generated_text):

        logits = model(encoded_input)
        logits = tf.squeeze(logits, 0)
        scaled_logits = logits * scale_factor
        new_char_index = tf.random.categorical(scaled_logits, num_samples=1)
        new_char_index = tf.squeeze(new_char_index)[-1].numpy()
        generated_string += str(character_array[new_char_index])
        new_char_index = tf.expand_dims([new_char_index], 0)
        encoded_input = tf.concat([encoded_input, new_char_index], axis=1)
        encoded_input = encoded_input[:, -max_input_length:]

    return generated_string


# Generate new text

tf.random.set_seed(1)
print(sample(lstm_model, starting_string='The island'))


# -------------------------------------------------------------------------------
# 5. GENERAL
# -------------------------------------------------------------------------------


# Show plots

plt.show()