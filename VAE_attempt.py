import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import Input, Dense, Reshape, Concatenate, Flatten, Lambda, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse
from tensorflow.keras.optimizers import Adam
import random
import os
from train import utils

def get_encoder_network(x, num_filters):
    x = Conv2D(num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    print("x shape", x.shape)
    x = Conv2D(num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    print("x shape", x.shape)
    x = MaxPooling2D()(x)
    print("x shape", x.shape)
    return x


def get_decoder_network(x, num_filters):
    x = UpSampling2D()(x)
    x = Conv2D(num_filters, 3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Conv2D(num_filters, 3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    return x


# function to create an autoencoder network
def get_vae(height, width, batch_size, latent_dim,
            is_variational=True, conditioning_dim=0,
            start_filters=8, nb_capacity=3,
            optimizer=Adam(lr=0.001)):
    # INPUT ##

    # create layer for input image
    # concatenate image metadata
    inputs = Input((height, width, 3))
    if conditioning_dim > 0:
        condition = Input([conditioning_dim])
        condition_up = Dense(height * width)(condition)
        condition_up = Reshape([height, width, 1])(condition_up)
        inputs_new = Concatenate(axis=3)([inputs, condition_up])
    else:
        inputs_new = inputs

    # ENCODER ##

    # create encoding layers
    # duplicate the encoding layers with increasing filters
    eblock = get_encoder_network(inputs_new, start_filters)
    for i in range(1, nb_capacity + 1):
        eblock = get_encoder_network(eblock, start_filters * (2 ** i))

    # create latent space layer
    _, *shape_spatial = eblock.get_shape().as_list()
    print("shape spatial", shape_spatial)
    eblock_flat = Flatten()(eblock)
    print("eblock_flat", eblock_flat.shape)
    if not is_variational:
        z = Dense(latent_dim)(eblock_flat)
    else:
        # sample latent values from a normal distribution
        def sampling(args):
            z_mean, z_log_sigma = args
            epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.)
            return z_mean + K.exp(z_log_sigma) * epsilon

        z_mean = Dense(latent_dim)(eblock_flat)
        z_log_sigma = Dense(latent_dim)(eblock_flat)
        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])
        print("z shape", z.shape)

    if conditioning_dim > 0:
        z_ext = Concatenate()([z, condition])

    ## DECODER ##

    # create decoding layers
    inputs_embedding = Input([latent_dim + conditioning_dim])
    embedding = Dense(np.prod(shape_spatial), activation='relu')(inputs_embedding)
    embedding = Reshape(eblock.shape.as_list()[1:])(embedding)
    print("embedding shape", embedding.shape)

    # duplicate the encoding layers with increasing filters
    dblock = get_decoder_network(embedding, start_filters * (2 ** nb_capacity))
    for i in range(nb_capacity - 1, -1, -1):
        dblock = get_decoder_network(dblock, start_filters * (2 ** i))

    print("dblock shape", dblock.shape)
    output = Conv2D(3, 1, activation='tanh')(dblock)

    ## VAE ##

    # put encoder, decoder together
    decoder = Model(inputs_embedding, output)
    if conditioning_dim > 0:
        encoder_with_sampling = Model(input=[inputs, condition], output=z)
        encoder_with_sampling_ext = Model(input=[inputs, condition], output=z_ext)
        vae_out = decoder(encoder_with_sampling_ext([inputs, condition]))
        vae = Model(input=[inputs, condition], output=vae_out)
    else:
        encoder_with_sampling = Model(inputs, z)
        vae_out = decoder(encoder_with_sampling(inputs))
        vae = Model(inputs, vae_out)

    # define the VAE loss as the sum of MSE and KL-divergence loss
    def vae_loss(x, x_decoded_mean):
        print("x", x.shape, "\n")
        print("x decoded mean", x_decoded_mean.shape)
        mse_loss = K.mean(mse(x, x_decoded_mean), axis=(1, 2)) * height * width
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
        return mse_loss + kl_loss

    if is_variational:
        vae.compile(loss=vae_loss, optimizer=optimizer)
    else:
        vae.compile(loss='mse', optimizer=optimizer)

    return vae, encoder_with_sampling, decoder


# hyperparameters
VARIATIONAL = False
HEIGHT = 32
WIDTH = 32
BATCH_SIZE = 21
LATENT_DIM = 21
START_FILTERS = 32
CAPACITY = 1
CONDITIONING = True
OPTIMIZER = Adam(lr=0.01)

vae, encoder, decoder = get_vae(is_variational=VARIATIONAL,
                                height=HEIGHT,
                                width=WIDTH,
                                batch_size=BATCH_SIZE,
                                latent_dim=LATENT_DIM,
                                conditioning_dim=0,  # hard code to zero for now
                                start_filters=START_FILTERS,
                                nb_capacity=CAPACITY,
                                optimizer=OPTIMIZER)


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

batch_size = 21  # Batch size for training.
epochs = 10  # Number of epochs to train for.
latent_dim = 21  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.

game = None
for path in replay_files:
    game = utils.HaliteV2(path)
    if game.game_play_list is not None and game.winner_id == 0:
        break
if game is None:
    print("get json")
    exit(0)

random_step = random.randint(1, 398)
game.prepare_data_for_vae()

X_ship = game.ship_position
Y_ship = game.ship_actions
halite_available = game.halite
my_shipyard = game.shipyard_position
my_cargo = game.cargo
gen = np.zeros(
    (399,32, 32, 3), # my ship, halite on map, my shipyard
    dtype='float32')
gen_val = np.zeros(
    (399, 32, 32, 1),
    dtype='float32')

pad_offset = 6

for i, (input_text, target_text) in enumerate(zip(X_ship, Y_ship)):

    # populate my ship presence no(index 0) or yes(index 1) = 1.
    for t, row in enumerate(input_text):
        for row_indx, item in enumerate(row):
            # print(count)
            gen[i, t+pad_offset, row_indx+pad_offset, 0] = item

    for t, row in enumerate(target_text):
        # print("t is", t)
        for row_indx, item in enumerate(row):
            # print("move is ", item, "row index is",  row_indx)
            gen_val[i, t+pad_offset, row_indx+pad_offset, 0] = item


for i, halite_map in enumerate(zip(halite_available)):
    # populate my ship presence no(index 0) or yes(index 1) = 1.
    # print("halite_map", halite_map)
    for t, row in enumerate(halite_map[0]):
        row = np.squeeze(row)
        # print("row is ", row)
        for row_indx, item in enumerate(row):
            # print(item)
            gen[i, t+pad_offset, row_indx+pad_offset, 1] = item

for i, shipyard_map in enumerate(zip(my_shipyard)):
    # populate my shipyard presence no(index 0) or yes(index 1) = 1.
    for t, row in enumerate(shipyard_map[0]):
        row = np.squeeze(row)
        # print("row is ", row)
        for row_indx, item in enumerate(row):
            # print(item)
            gen[i, t+pad_offset, row_indx+pad_offset, 2] = item

print("gen shape", gen.shape)
print("gen val", gen_val.shape)

# train the variational autoencoder
vae.fit(x=gen, y=gen_val,
          batch_size=BATCH_SIZE,
          epochs=10,
          validation_split=0.2)

vae.save('vae.h5')