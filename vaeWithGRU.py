import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

from tensorflow.python.keras.layers import Dense

from train import utils

"""
## Create a sampling layer
"""

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

seq_list = []
game = None
training_datasets = []
for i, path in enumerate(replay_files):
    game = utils.HaliteV2(path)
    print("index", i)
    if i == 1:
        break
    if game.game_play_list is not None and game.winner_id == 0:
        game.prepare_data_for_vae()
        """
        Four features as training input:
            1) halite available
            2) my ship
            3) cargo on my ship
            4) my shipyard
        """
        training_input = np.zeros(
            (400, 32, 32, 4),
            dtype='float32')

        my_ship_positions = game.ship_position
        target_ship_actions = game.ship_actions
        halite_available = game.halite
        my_shipyard = game.shipyard_position
        my_cargo = game.cargo

        """
        Target ship actions:
        """
        training_label = np.zeros(
            (400, 32, 32, 6),
            dtype='float32')

        pad_offset = 6

        #  1) halite available
        for i, halite_map in enumerate(zip(halite_available)):
            # print("halite_map", halite_map)
            for row_indx, row in enumerate(halite_map[0]):
                row = np.squeeze(row)
                for col_indx, item in enumerate(row):
                    # print(item)
                    training_input[i, row_indx + pad_offset, col_indx + pad_offset, 0] = item * 10

        # 2) my ship position
        for i, my_ship_position in enumerate(my_ship_positions):
            for row_indx, row in enumerate(my_ship_position):
                for col_indx, item in enumerate(row):
                    training_input[i, row_indx + pad_offset, col_indx + pad_offset, 1] = item * 10

        # 3) cargo on my ship
        for i, cargo_map in enumerate(my_cargo):
            for row_indx, row in enumerate(cargo_map):
                for col_indx, item in enumerate(row):
                    training_input[i, row_indx + pad_offset, col_indx + pad_offset, 2] = item * 10

        # 4) my ship yard position
        for i, shipyard_map in enumerate(my_shipyard):
            for row_indx, row in enumerate(shipyard_map):
                for col_indx, item in enumerate(row):
                    training_input[i, row_indx + pad_offset, col_indx + pad_offset, 3] = item * 10

        # target actions
        for i, target_ship_action in enumerate(target_ship_actions):
            for row_indx, row in enumerate(target_ship_action):
                for col_indx, item in enumerate(row):
                    training_label[i, row_indx + pad_offset, col_indx + pad_offset, int(item)] = 1.

        print("training input shape", training_input.shape)
        print("target action shape", training_label.shape)
        board_size = game.config["size"]
        vocab_dict = {}
        num_dict = {}
        for i in range(board_size ** 2):
            vocab_dict[str(i)] = i
            num_dict[i] = str(i)
        vocab_idx = board_size ** 2
        move_option = ["EAST", "WEST", "SOUTH", "NORTH", "CONVERT", "SPAWN", "NO", "(", ")"]
        for option in move_option:
            vocab_dict[option] = vocab_idx
            num_dict[vocab_idx] = option
            vocab_idx += 1
        # target actions
        decoder_input_data = np.zeros(
            (400, 50, len(vocab_dict)),
            dtype=np.float32)
        decoder_target_data = np.zeros(
            (400, 50, len(vocab_dict)),
            dtype=np.float32)

        sequence = game.move_sequence
        sequence.append(sequence[-1])
        seq_list.append(sequence[-1])
        # TODO: validate max sequence
        for step, each_sequence in enumerate(sequence):
            each_sequence_list = each_sequence.split()
            seq_list.append(each_sequence)
            idx = 0
            last_word = ""
            for each_word in each_sequence_list:
                assert (each_word in vocab_dict)
                assert (idx < 50)
                if idx == 49:
                    break
                if idx == 0:
                    decoder_input_data[step][idx][448] = 1.
                    decoder_target_data[step][idx][vocab_dict[each_word]] = 1.
                else:
                    decoder_input_data[step][idx][vocab_dict[last_word]] = 1.
                    decoder_target_data[step][idx][vocab_dict[each_word]] = 1.
                idx += 1
                last_word = each_word
            decoder_input_data[step][idx][vocab_dict[last_word]] = 1.
            decoder_target_data[step][idx][449] = 1.
        print("target action shape", decoder_target_data.shape)
       # train_dataset = tf.data.Dataset.from_tensor_slices((training_input, training_label))

      # print("dataset shape", len(list(train_dataset.as_numpy_iterator())))
        train_dataset = [[training_input, decoder_input_data],
                         decoder_input_data]
        training_datasets.append(train_dataset)

train_dataset = training_datasets[0]

# for i in range(1, len(training_datasets)):
#     train_dataset = train_dataset.concatenate((training_datasets[i]))


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


"""
## Build the encoder
"""

latent_dim = 2

encoder_inputs = keras.Input(shape=(32, 32, 4))
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

"""
## Build the decoder
"""

# latent_inputs = keras.Input(shape=(latent_dim,))
# x = layers.Dense(8 * 8 * 64, activation="relu")(latent_inputs)
# x = layers.Reshape((8, 8, 64))(x)
# x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
# x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
# decoder_outputs = layers.Conv2DTranspose(6, 3, activation="sigmoid", padding="same")(x)
# decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
# decoder.summary()
decoder_input = keras.Input(shape=(None, 450))
gru = tf.keras.layers.GRU(latent_dim, return_sequences=True, return_state=True)
decoder_state_input_h = keras.Input(shape=(latent_dim,))
decoder_outputs, state_h = gru(decoder_input, initial_state=decoder_state_input_h)
decoder_outputs = Dense(450, activation='softmax')(decoder_outputs)
print("decoder output shape: ", decoder_outputs.shape)
decoder = keras.Model(
        [decoder_input, decoder_state_input_h],
        [decoder_outputs, state_h])
decoder.summary()
"""
## Define the VAE as a `Model` with a custom `train_step`
"""


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def decode_sequence(self, input_seq):
        # Encode the input as state vectors.
        z_mean, z_log_var, z = self.encoder(input_seq)
        states_value = z
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, 450))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, 448] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h = self.decoder(
                [target_seq, states_value])

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = num_dict[int(sampled_token_index)]
            decoded_sentence += sampled_char
            decoded_sentence += " "
            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == ')' or
                    len(decoded_sentence) > 49):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, 450))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = h
        return decoded_sentence
    @tf.function
    # def call(self, inputs, training=False):
    #     z_mean, z_log_var, z = self.encoder(inputs)
    #     reconstruction = self.decoder(z)
    #     return reconstruction
    def call(self, inputs, training=False):
        #z_mean, z_log_var, z = self.encoder(inputs)
        print(inputs)
        inputs = np.array(inputs)
        reconstruction = self.decode_sequence(inputs)#self.decoder(z)
        return reconstruction
    def train_step(self, data):
        print("data is", data)
       # if isinstance(data, list):
        print("im here")

        x = data[0][0]
        y1 = data[0][1]
        y2 = data[0][2]
        with tf.GradientTape() as tape:
            # print("data shape", data.shape)
            # print("y shape", y.shape)
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction, _ = self.decoder([y1, z])
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(y2, reconstruction)
            )
            reconstruction_loss *= 32 * 32
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }
    # def train_step(self, data):
    #     print("data is", data)
    #     y = None
    #     if isinstance(data, tuple):
    #         print("im here")
    #         y = data[1]
    #         data = data[0]
    #     with tf.GradientTape() as tape:
    #         print("data shape", data.shape)
    #         print("y shape", y.shape)
    #         z_mean, z_log_var, z = self.encoder(data)
    #         reconstruction = self.decoder(z)
    #         reconstruction_loss = tf.reduce_mean(
    #             keras.losses.binary_crossentropy(y, reconstruction)
    #         )
    #         reconstruction_loss *= 32 * 32
    #         kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    #         kl_loss = tf.reduce_mean(kl_loss)
    #         kl_loss *= -0.5
    #         total_loss = reconstruction_loss + kl_loss
    #     grads = tape.gradient(total_loss, self.trainable_weights)
    #     self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
    #     return {
    #         "loss": total_loss,
    #         "reconstruction_loss": reconstruction_loss,
    #         "kl_loss": kl_loss,
    #     }


"""
## Train the VAE
"""
for i in range(1):
    train_x = np.empty((400, 32, 32, 4))
    train_y_1 = np.empty((400, 50, 450))
    train_y_2 = np.empty((400, 50, 450))
    random_idx = []
    for j in range(400):
        random_idx.append([random.randint(0, len(training_datasets) - 1), random.randint(0, 399)])
    for idx, temp in enumerate(random_idx):
        # Find list of IDs
        training_training = training_datasets[temp[0]]
        train_x[idx,] = training_training[0][0][temp[1]]
        train_y_1[idx,] = training_training[0][1][temp[1]]
        train_y_2[idx,] = training_training[1][temp[1]]
#train_dataset = train_dataset.shuffle(7200, reshuffle_each_iteration=True).batch(40)

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(lr=0.005))
    #vae.fit(train_dataset, epochs=10)
    vae.fit([train_x, train_y_1, train_y_2], epochs=3)


def decode_sequence( input_seq):
    # Encode the input as state vectors.
    z_mean, z_log_var, z = vae.encoder(input_seq)
    states_value = z
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, 450))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, 448] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h = vae.decoder(
            [target_seq, states_value])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = num_dict[int(sampled_token_index)]
        decoded_sentence += sampled_char
        decoded_sentence += " "
        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == ')' or
                len(decoded_sentence) > 49):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, 450))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = h
    return decoded_sentence
test = np.zeros(
            (1, 32, 32, 4),
            dtype='float32')

print(decode_sequence(test))
tf.saved_model.save(vae, 'bot/vae_new/')
#vae.save('bot/vae_new', save_format="tf")
print("done")