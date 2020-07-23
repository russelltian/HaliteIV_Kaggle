import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
import numpy as np
import random
import os
from train import utils

# Define an input sequence and process it.
num_encoder_tokens = 2 # either yes or no
latent_dim = 100 # hard code
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
num_decoder_tokens = 6 #
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


"""Data Extraction"""

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
latent_dim = 100  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.

path = replay_files[0]
# Load all training data
game = utils.Halite()
game.load_replay(path)
game.load_data()

random_step = random.randint(1, 398)
X_ship = game.ship_position
Y_ship = game.ship_actions
halite_available = game.halite
my_shipyard = game.shipyard_position
my_cargo = game.cargo
encoder_input_data = np.zeros(
    (399, 1764, 2), # my ship, halite on map, my shipyard, my cargo
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
    # populate my ship presence no(index 0) or yes(index 1) = 1.
    for t, row in enumerate(input_text):
        for item in row:
            # print(count)
            encoder_input_data[i, count, int(item)] = 1.
            count += 1
    count = 0
    for t, row in enumerate(target_text):
        for item in row:
            decoder_input_data[i, count, int(item)] = 1.
            if count > 0:
                decoder_target_data[i, count - 1,int(item)] = 1.
            count += 1

for i, halite_map in enumerate(zip(halite_available)):
    count = 441
    # populate my ship presence no(index 0) or yes(index 1) = 1.
    # print("halite_map", halite_map)
    for t, row in enumerate(halite_map[0]):
        row = np.squeeze(row)
        # print("row is ", row)
        for item in row:
            # print(item)
            if item > 0.01:
                encoder_input_data[i, count, 0] = 1.
            else:
                encoder_input_data[i, count, 1] = 1.
            count += 1

for i, (shipyard_map) in enumerate(zip(my_shipyard)):
    count = 882
    # populate my shipyard presence no(index 0) or yes(index 1) = 1.
    for t, row in enumerate(shipyard_map[0]):
        # print("row is", row)
        for item in row:
            # print(count)
            encoder_input_data[i, count, int(item)] = 1.
            count += 1

for i, (cargo_map) in enumerate(zip(my_cargo)):
    count = 882
    # populate my cargo amount <=0.01(index 0) or >0.01(index 1) = 1.
    for t, row in enumerate(cargo_map[0]):
        row = np.squeeze(row)
        # print("row is ", row)
        for item in row:
            # print(item)
            if item > 0.01:
                encoder_input_data[i, count, 0] = 1.
            else:
                encoder_input_data[i, count, 1] = 1.
            count += 1


# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)

model.save('s2s.h5')