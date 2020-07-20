import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
import numpy as np
import random
import os
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
epochs = 5  # Number of epochs to train for.
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
    for t, row in enumerate(input_text):
        for item in row:
            # print(count)
            encoder_input_data[i, count, int(item)] = 1.
    for t, row in enumerate(target_text):
        for item in row:
            decoder_input_data[i, count, int(item)] = 1.
            if count > 0:
                decoder_target_data[i, count - 1,int(item)] = 1.
            count += 1
# Define an input sequence and process it
encoder_inputs = Input(shape=(None, 6))
encoder = LSTM(latent_dim,return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]
num_decoder_tokens = 6
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
# Save model
model.save('s2s.h5')