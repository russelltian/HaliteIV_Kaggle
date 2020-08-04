import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
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


game = None
training_datasets = []
for i, path in enumerate(replay_files):
    game = utils.HaliteV2(path)
    print("index", i)
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

        train_dataset = tf.data.Dataset.from_tensor_slices((training_input, training_label))

        print("dataset shape", len(list(train_dataset.as_numpy_iterator())))

        training_datasets.append(train_dataset)

train_dataset = training_datasets[0]

print(train_dataset)

for i in range(1, len(training_datasets)):
    train_dataset = train_dataset.concatenate((training_datasets[i]))


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

latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(8 * 8 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((8, 8, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(6, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

"""
## Define the VAE as a `Model` with a custom `train_step`
"""


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    @tf.function
    def call(self, inputs, training=False):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction

    def train_step(self, data):
        print("data is", data)
        y = None
        if isinstance(data, tuple):
            print("im here")
            y = data[1]
            data = data[0]
        with tf.GradientTape() as tape:
            print("data shape", data.shape)
            print("y shape", y.shape)
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(y, reconstruction)
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


"""
## Train the VAE
"""
train_dataset = train_dataset.shuffle(7200, reshuffle_each_iteration=True).batch(40)


vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(lr=0.005))
vae.fit(train_dataset, epochs=10)

test =  np.zeros(
            (1, 32, 32, 4),
            dtype='float32')
vae.predict(test)
tf.saved_model.save(vae, 'bot/vae_new/')
#vae.save('bot/vae_new', save_format="tf")
print("done")