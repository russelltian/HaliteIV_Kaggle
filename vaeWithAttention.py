import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from train import utils
from multiprocessing import Pool

training_datasets = []
vocab_size = 450
MAX_WORD_LENGTH = 50
units = 512
embedding_dim = 256
FEATURE_MAP_DIMENSION = 5 # TRAINING INPUT dimension
inference_decoder = utils.Inference(board_size=21)
BATCH_SIZE = 100
DATASET_SIZE = 400
METADATA_DIM = 3 # my halite amount, turns left, most leading opponent halite amount

"""
    Data Extraction
"""
PATH = 'train/test_replay/'
replay_files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(PATH):
    for file in f:
        if '.json' in file:
            replay_files.append(os.path.join(r, file))
for f in replay_files:
    print(f)


def load_raw_data(path):
    # for i, path in enumerate(replay_files):
    game = utils.HaliteV2(path)
    print("loading file from ", path)
    if game.game_play_list is not None:
        """
        Five features as training input:
            1) halite available
            2) my ship
            3) cargo on my ship
            4) my shipyard
            5) other players' ships

        Sequence is a string that record the actions of ships/shipyards
        """
        training_input, meta_data, sequence = game.prepare_vae_encoder_input()

        # target actions
        assert (inference_decoder.dictionary_size == 450)
        decoder_input_sequence = []
        decoder_target_sequence = []
        # TODO: validate max sequence
        for step, each_sequence in enumerate(sequence):
            # Add ( and ) for teacher forcing
            input_sequence = '( ' + each_sequence
            output_sequence = each_sequence + ')'
            decoder_input_sequence.append(input_sequence)
            decoder_target_sequence.append(output_sequence)
        assert (len(decoder_target_sequence) == len(decoder_input_sequence) == 400)

        train_dataset = [training_input, meta_data, decoder_input_sequence,
                         decoder_target_sequence]

        training_datasets.append(train_dataset)


# pool = Pool()
# pool.map(load_raw_data, replay_files)
for i in range(len(replay_files)):
    load_raw_data(replay_files[i])

"""
    A note to myself:
        train_dataset[0] -> training_input:
                                np array, shape: (400, 1, 32, 32, 5)
        train_dataset[1] -> decoder_input_sequence:
                            len(400)
                            each element varies in length
                            each element does not have EOS ")"
"""
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

"""
    CNN Encoder
"""

encoder_inputs = keras.Input(shape=(32, 32, FEATURE_MAP_DIMENSION,))
meta_data = keras.Input(shape=(METADATA_DIM))
condition_up = layers.Dense(32 * 32)(meta_data)
condition_up = layers.Reshape(target_shape=(32, 32, 1))(condition_up)
encoder_inputs_new = layers.Concatenate(axis=3)([encoder_inputs, condition_up]) # concatenate meta data
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs_new)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Reshape(target_shape=(64, 64))(x)
z = layers.Dense(embedding_dim, activation="relu")(x)
encoder = keras.Model([encoder_inputs, meta_data], z, name="encoder")
# x = layers.Dense(128, activation="relu")(x)
# z_mean = layers.Dense(embedding_dim, name="z_mean")(x)
# z_log_var = layers.Dense(embedding_dim, name="z_log_var")(x)
# z = Sampling()([z_mean, z_log_var])
# encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

"""
    GRU Decoder
"""

decoder_inputs = keras.Input(shape=(1))
features = keras.Input(shape=(64, embedding_dim))
hidden = keras.Input(shape=(units))
# add attention below ========
hidden_with_time_axis = tf.expand_dims(hidden, 1)
w1 = layers.Dense(units)(features)
w2 = layers.Dense(units)(hidden_with_time_axis)
score = tf.nn.tanh(w1 + w2)
V = layers.Dense(1)(score)
attention_weights = tf.nn.softmax(V, axis=1)
context_vector = attention_weights * features
context_vector = tf.reduce_sum(context_vector, axis=1)
# =========
embedding = layers.Embedding(vocab_size, embedding_dim)
x = embedding(decoder_inputs)
x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
gru = layers.GRU(units,
                           return_sequences=True,
                           return_state=True,
                           recurrent_initializer='glorot_uniform')
output, state = gru(x)
x = layers.Dense(units)(output)
x = tf.reshape(x, (-1, x.shape[2]))
x = layers.Dense(vocab_size)(x)
decoder = keras.Model(
    [decoder_inputs, features, hidden],
    [x, state, attention_weights], name="decoder_with_attention"
)
decoder.summary()


ONE_HOT_WORD_LENGTH = 450
EPOCHS = 10

"""
    both decoder_input and decoder_target shapes are (BATCH_SIZE, MAX_WORD_LENGTH)
        i.e. (40, 50)
"""

class VAEwithAttention(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAEwithAttention, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    @tf.function
    def call(self, inputs, training=False):
        # z_mean, z_log_var, features = self.encoder(inputs)
        features = self.encoder(inputs)
        return features

    def train_step(self, data):
        input_image = data[0][0]
        meta_data = data[0][1]
        # print(input_image.shape)
        # print(data[0])
        decoder_target = data[0][2]
        # print(decoder_target.shape)
        # print("tf shape", tf.shape(input_image))
        #input_image = tf.reshape(input_image, shape=(BATCH_SIZE, 32, 32, 5))
        # print(input_image)
        # print(input_image.shape)
        #input_image = tf.ones(shape=(BATCH_SIZE, 32, 32, 5))

        def loss_function(real, pred):
            # mask = tf.math.logical_not(tf.math.equal(real, 0))
            # print("mask", mask)

            loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction='none')

            loss_ = loss_object(real, pred)

            # mask = tf.cast(mask, dtype=loss_.dtype)
            # loss_ *= mask

            return tf.reduce_mean(loss_)

        loss = 0.0

        with tf.GradientTape() as tape:
            hidden = tf.zeros((BATCH_SIZE, units))
            dec_input = tf.expand_dims([float(inference_decoder.word_to_index_mapping['('])] * BATCH_SIZE, 1)
            features = self.encoder([input_image, meta_data])
            #features = tf.ones(shape=(BATCH_SIZE, 64, 128))
            # print("features", features)
            # print("dec_input", dec_input)
            for i in range(0, decoder_target.shape[1]):
                predictions, hidden, _ = self.decoder([dec_input, features, hidden])
                dec_input = tf.expand_dims(decoder_target[:, i], 1)
                loss += loss_function(decoder_target[:, i], predictions)

        trainable_variables = self.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        return {"loss": loss}


"""
    Training
"""

print("create vae\n")
vae = VAEwithAttention(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(lr=0.004))

#     total_loss = 0
# for batch in range(1):
for file in range(len(replay_files)):
    train_x = np.empty((BATCH_SIZE, 32, 32, 5))
    train_y_2 = np.empty((BATCH_SIZE, MAX_WORD_LENGTH))
    train_x_meta = np.empty(shape=(BATCH_SIZE, METADATA_DIM))
    random_idx = []
    for j in range(BATCH_SIZE):
        #random_idx.append([random.randint(0, len(training_datasets) - 1), random.randint(0, 399)])
        random_idx.append([file,  random.randint(0, 399)])
    for idx, choice in enumerate(random_idx):
        # choice[0] is which gameplay we pick
        # choice[1] is which step in the gameplay we pick
        # Find list of IDs
        each_training_sample = training_datasets[choice[0]]
        train_x[idx,] = each_training_sample[0][choice[1]]
        train_x_meta[idx,] = each_training_sample[1][choice[1]]
        input_sequence = each_training_sample[2][choice[1]]
        output_sequence = each_training_sample[3][choice[1]]

        # convert sequence to vector
        decoder_input_data = np.zeros(
            (MAX_WORD_LENGTH),
            dtype=np.float32) # 1, 50
        decoder_target_data = np.zeros(
            (MAX_WORD_LENGTH),
            dtype=np.float32)

        input_sequence_list = input_sequence.split()
        output_sequence_list = output_sequence.split()
        assert (len(input_sequence_list) == len(output_sequence_list))
        for word_idx in range(len(output_sequence_list)):
            output_word = output_sequence_list[word_idx]
            # print("input word", input_word, "output word", output_word)
            # if input_word == '(' or output_word == ')':
            #     print(input_word, output_word)
            # TODO : increase length of sentence
            decoder_target_data[word_idx] = inference_decoder.word_to_index_mapping[output_word]
            if word_idx == MAX_WORD_LENGTH - 1:
                break
            # print(decoder_target_data[word_idx])
        train_y_2[idx ] = decoder_target_data

    vae.fit([train_x, train_x_meta, train_y_2], epochs=3, verbose=2, batch_size=BATCH_SIZE)


# batch_loss, t_loss = train_step(train_x, train_y_1, train_y_2)
# total_loss += t_loss
# print('Epoch {} Loss {:.6f}'.format(epoch + 1,
#                                     total_loss))
print("saving model")
tf.saved_model.save(vae, 'bot/vae_attention')

print("finished training")