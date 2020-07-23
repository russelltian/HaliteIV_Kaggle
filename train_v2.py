import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
import numpy as np
import random
import os

from tensorflow.python.keras.layers import GRU

from train import utils

PATH = 'train/top_replay/'
replay_files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(PATH):
    for file in f:
        if '.json' in file:
            replay_files.append(os.path.join(r, file))
for f in replay_files:
    print(f)

batch_size = 16  # Batch size for training.
epochs = 10  # Number of epochs to train for.
latent_dim = 32  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.

path = replay_files[0]
# Load all training data
game = utils.Halite()
game.load_replay(path)
game.load_data()

random_step = random.randint(1, 398)
X_ship = game.ship_position
Y_ship = game.ship_actions
encoder_input_data = np.zeros(
    (399, 441, 6),
    dtype='float32')
decoder_input_data = np.zeros(
    (399, 441, 6),
    dtype='float32')
decoder_target_data = np.zeros(
    (399, 441, 6),
    dtype='float32')
print(X_ship.shape, Y_ship.shape)

for i, (input_text, target_text) in enumerate(zip(X_ship, Y_ship)):
    count = 0
    count1 = 0
    for t, row in enumerate(input_text):
        for item in row:
            # print(count)
            encoder_input_data[i, count, int(item)] = 1.
            count += 1
    for t, row in enumerate(target_text):
        for item in row:
            decoder_input_data[i, count1, int(item)] = 1.
            if count1 > 0:
                decoder_target_data[i, count1 - 1,int(item)] = 1.
            count1 += 1
# Define an input sequence and process it
num_decoder_tokens = 6
num_encoder_tokens = 6
# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


# encoder_outputs, state_h = encoder(encoder_inputs)
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)

encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

encoder_model.save('encoder.h5')
decoder_model.save('decoder.h5')
print(model.summary())
# Save model
model.save('s2s.h5')