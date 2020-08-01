import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import Input, Dense, Concatenate, Flatten, Lambda, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse
from tensorflow.keras.optimizers import Adam
import os

from tensorflow.python.keras.layers import GRU
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
    x = Conv2D(num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2D(num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    return x


# function to create an autoencoder network
def get_vae(height, width, batch_size, latent_dim,
            is_variational=True, conditioning_dim=0,
            start_filters=8, nb_capacity=3,
            optimizer=Adam(lr=0.001)):
    # INPUT ##

    # create layer for input image
    # concatenate image metadata
    inputs = Input((height, width, 4))
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
        print("z shape", z.shape)
    else:
        # sample latent values from a normal distribution
        def sampling(args):
            z_mean, z_log_sigma = args
            epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.)
            print("epsilon shape", epsilon.shape)
            return z_mean + K.exp(z_log_sigma) * epsilon

        z_mean = Dense(latent_dim)(eblock_flat)
        print("z_mean shape", z_mean.shape)
        z_log_sigma = Dense(latent_dim)(eblock_flat)
        print("z_log_sigma shape", z_log_sigma.shape)
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
    # dblock = get_decoder_network(embedding, start_filters * (2 ** nb_capacity))
    # for i in range(nb_capacity - 1, -1, -1):
    #     dblock = get_decoder_network(dblock, start_filters * (2 ** i))
    #
    # print("dblock shape", dblock.shape)
    # output = Conv2D(6, 1, activation='tanh')(dblock)
    # print("output shape", output.shape)

    ## VAE ##
    # put encoder, decoder together
    # decoder = Model(inputs_embedding, output)
    decoder_input = Input(shape=(None, 450))
    gru = GRU(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _ = gru(decoder_input, initial_state=z)
    decoder_outputs = Dense(450, activation='softmax')(decoder_outputs)
    print("decoder output shape: ", decoder_outputs.shape)
    if conditioning_dim > 0:
        encoder_with_sampling = Model(input=[inputs, condition], output=z)
        # encoder_with_sampling_ext = Model(input=[inputs, condition], output=z_ext)
        # vae_out = decoder(encoder_with_sampling_ext([inputs, condition]))
        # vae = Model(input=[inputs, condition], output=vae_out)
    else:
        vae = Model([inputs, decoder_input], decoder_outputs)
    encoder_model = Model(inputs, z)
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_outputs, state_h = gru(decoder_input, initial_state=decoder_state_input_h)
    decoder_outputs = Dense(450, activation='softmax')(decoder_outputs)
    decoder_model = Model(
        [decoder_input, decoder_state_input_h],
        [decoder_outputs, state_h])

    # define the VAE loss as the sum of MSE and KL-divergence loss
    def vae_loss(vae_out, dupout):
        print("x", vae_out.shape)
        print("x is", vae_out)
        print("x decoded mean", dupout.shape)
        print("x decoded mean is", dupout)
        print("mse is", mse(vae_out, dupout))
        # mse_loss = K.mean(mse(vae_out, dupout), axis=(1, 2)) * height * width
        # print("mse_loss shape", mse_loss.shape)
        # print("z log sigma", K.exp(z_log_sigma))
        # print("z mean", K.square(z_mean))
        # print("hi", 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
        # kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
        # print("kl_loss shape", kl_loss.shape)
        # print("here", kl_loss)
        # print("whats up")
        return mse(vae_out, dupout)

    if is_variational:
        # vae.compile(loss=vae_loss, optimizer=optimizer)
        vae.compile(optimizer=optimizer, loss='categorical_crossentropy',
                    metrics=['accuracy'])
    else:
        # vae.compile(loss='mse', optimizer=optimizer)
        vae.compile(optimizer=optimizer, loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return vae, encoder_model, decoder_model
    # return vae, encoder_with_sampling, decoder


# hyperparameters
VARIATIONAL = False
HEIGHT = 32
WIDTH = 32
BATCH_SIZE = 40
LATENT_DIM = 8
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

game = None
lot_of_game = utils.LoadGame()
for path in replay_files:
    game = utils.HaliteV2(path)
    assert(game.game_play_list is not None and game.winner_id == 0)
    game.prepare_data_for_vae()
    break
if game is None:
    print("get json")
    exit(0)


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
target_action = np.zeros(
    (400, 32, 32, 6),
    dtype='float32')

pad_offset = 6

#  1) halite available
for i, halite_map in enumerate(zip(halite_available)):
    # print("halite_map", halite_map)
    for row_indx, row in enumerate(halite_map[0]):
        row = np.squeeze(row)
        for col_indx, item in enumerate(row):
            training_input[i, row_indx + pad_offset, col_indx + pad_offset, 0] = item * 10
# 2) my ship position
for i, my_ship_position in enumerate(my_ship_positions):
    for row_indx, row in enumerate(my_ship_position):
        for col_indx, item in enumerate(row):
            training_input[i, row_indx + pad_offset, col_indx + pad_offset, 1] = item * 50
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
# Do word embedding
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
    (400, 441, len(vocab_dict)),
    dtype='float32')
decoder_target_data = np.zeros(
    (400, 441, len(vocab_dict)),
    dtype='float32')

sequence = game.move_sequence
sequence.append(sequence[-1])

# TODO: validate max sequence
for step, each_sequence in enumerate(sequence):
    each_sequence_list = each_sequence.split()
    idx = 0
    last_word = ""
    for each_word in each_sequence_list:
        assert (each_word in vocab_dict)
        assert (idx < 441)
        if idx == 0:
            decoder_input_data[step][idx][-2] = 1.
            decoder_target_data[step][idx][vocab_dict[each_word]] = 1.
        else:
            decoder_input_data[step][idx][vocab_dict[last_word]] = 1.
            decoder_target_data[step][idx][vocab_dict[each_word]] = 1.
        idx += 1
        last_word = each_word
    decoder_input_data[step][idx][vocab_dict[last_word]] = 1.
    decoder_target_data[step][idx][-1] = 1.
# for i, target_ship_action in enumerate(target_ship_actions):
#     for row_indx, row in enumerate(target_ship_action):
#         for col_indx, item in enumerate(row):
#             target_action[i, row_indx + pad_offset, col_indx + pad_offset, int(item)] = 1.
#             decoder_input_data[i][row_indx * 21 + col_indx][item] = 1.
#             decoder_target_data[i][row_indx * 21 + col_indx-1][item] = 1.
# decoder_target_data = np.array(decoder_target_data)
# print(decoder_target_data.shape, training_input.shape)
print("training input shape", training_input.shape)
print("target action shape", decoder_target_data.shape)

# train the variational autoencoder
vae.fit(x=[training_input, decoder_input_data], y=decoder_target_data,
        batch_size=BATCH_SIZE,
        epochs=20,
        validation_split=0.2)
result = vae.predict([training_input, decoder_input_data])
# for step, i in enumerate(result):
#     for position, j in enumerate(i):
#         if np.argmax(j) != 0:
#             print("at step: ", step, ", item at position ", position, " is ", np.argmax(j))
encoder.save('bot/vae_encoder.h5')
decoder.save('bot/vae_decoder.h5')


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, 450))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, 448] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h = decoder.predict(
            [target_seq, states_value])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = num_dict[int(sampled_token_index)]
        decoded_sentence += sampled_char
        decoded_sentence += " "
        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == ')' or
                len(decoded_sentence) > 441):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, 450))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = h

    return decoded_sentence


for i in range(10):
    print(decode_sequence(training_input[i:i + 1, :, :, :]))
print("end of training ")
