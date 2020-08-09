import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from tensorflow.python.keras.layers import Dense
from train import utils
from multiprocessing import Pool, Process, Manager

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

manager = Manager()
training_datasets = manager.list()
ONE_HOT_WORD_LENGTH = 450
MAX_WORD_LENGTH = 50
latent_dim = 16
FEATURE_MAP_DIMENSION = 5 # TRAINING INPUT dimension
inference_decoder = utils.Inference(board_size=21)
BATCH_SIZE = 40
embedding_dim = 64
game = None
jhh = np.empty((BATCH_SIZE, ONE_HOT_WORD_LENGTH))
a = keras.layers.Embedding(ONE_HOT_WORD_LENGTH,64)
print(a(jhh).shape)
def load_raw_data(path, i, training_datasets):
    # for i, path in enumerate(replay_files):
        print("file idx :", i)
        game = utils.HaliteV2(path)
        print("loading file from ", path)
        if game.game_play_list is not None:
            """
            Four features as training input:
                1) halite available
                2) my ship
                3) cargo on my ship
                4) my shipyard
                5) other players' ships
                
            Sequence is a string that record the actions of ships/shipyards
            """
            training_input, sequence = game.prepare_vae_encoder_input()
            """
            Target ship actions:
            """

            pad_offset = 6
            board_size = game.config["size"]

            # target actions
            assert(inference_decoder.dictionary_size == 450)
            decoder_input_sequence = []
            decoder_target_sequence = []
            # TODO: validate max sequence
            for step, each_sequence in enumerate(sequence):
                # Add ( and ) for teacher forcing
                input_sequence = '( ' + each_sequence
                output_sequence = each_sequence + ')'
                decoder_input_sequence.append(input_sequence)
                decoder_target_sequence.append(output_sequence)
            assert(len(decoder_target_sequence) == len(decoder_input_sequence) == 400)

            train_dataset = [training_input, decoder_input_sequence,
                             decoder_target_sequence]
            training_datasets.append(train_dataset)
process_list = []
for i in range(2):
    p = Process(target=load_raw_data, args=(replay_files[i], i, training_datasets))
    p.start()
    process_list.append(p)
for i in process_list:
    p.join()
print(len(training_datasets))

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
class VAE_CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, latent_dim):
        super(VAE_CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.conv1 = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")
        self.conv2 = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(16, activation="relu")
        self.z_mean = layers.Dense(latent_dim, name="z_mean")
        self.z_log_var = layers.Dense(latent_dim, name="z_log_var")
        self.z = Sampling()

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        embedding = layers.Reshape((-1,64))(x)
        embedding = tf.nn.relu(embedding)
        # batch * 64 * 64
        x = self.flatten(x)
        x = self.dense1(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.z([z_mean, z_log_var])
        return embedding, z, z_mean, z_log_var

# Bahdanau is one variant of the attention mechanism.
class BahdanauAttention(tf.keras.Model):
  def __init__(self, latent_dim):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(latent_dim)
    self.W2 = tf.keras.layers.Dense(latent_dim)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    # features(CNN_encoder output) shape == (batch_size, ?, latent_dim)

    # hidden shape == (batch_size, latent_dim)
    # hidden_with_time_axis shape == (batch_size, 1, latent_dim)
    hidden_with_time_axis = tf.expand_dims(hidden, 1)
    # score shape == (batch_size, 64, latent_dim)
    score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
     #   (BATCH_SIZE, 64, latent_dim) + (BATCH_SIZE, 1, latent_dim)
    # attention_weights shape == (batch_size, 64, 1)
    # You get 1 at the last axis because you are applying score to self.V

   # print("scoreshape", score.shape)

    attention_weights = tf.nn.softmax(self.V(score), axis=1)
  #  print("attention_weights shape", attention_weights.shape)
    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights
# encoder_inputs = keras.Input(shape=(32, 32, FEATURE_MAP_DIMENSION))
# x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
# x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
# x = layers.Flatten()(x)
# x = layers.Dense(16, activation="relu")(x)
# z_mean = layers.Dense(latent_dim, name="z_mean")(x)
# z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
# z = Sampling()([z_mean, z_log_var])
# encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
# encoder.summary()

"""
## Build the decoder
"""
class GRU_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, latent_dim, vocab_size):
        super(GRU_Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.fc0 = tf.keras.layers.Dense(embedding_dim)
        self.gru = tf.keras.layers.GRU(self.latent_dim, return_sequences=True,
                                       return_state=True, recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.latent_dim)
        self.fc2 = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.latent_dim)

    def call(self, x, features, hidden):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)
      #  print(context_vector.shape, attention_weights.shape)
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = tf.expand_dims(self.fc0(x), 1)
      #  print("decoder after dense x shape ", x.shape)
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.latent_dim))

# decoder_input = keras.Input(shape=(None, ONE_HOT_WORD_LENGTH))
# gru = tf.keras.layers.GRU(latent_dim, return_sequences=True, return_state=True)
# decoder_state_input_h = keras.Input(shape=(latent_dim,))
# decoder_outputs, state_h = gru(decoder_input, initial_state=decoder_state_input_h)
# decoder_outputs = Dense(ONE_HOT_WORD_LENGTH, activation='softmax')(decoder_outputs)
# print("decoder output shape: ", decoder_outputs.shape)
# decoder = keras.Model(
#         [decoder_input, decoder_state_input_h],
#         [decoder_outputs, state_h])
# decoder.summary()


encoder = VAE_CNN_Encoder(latent_dim)
decoder = GRU_Decoder(embedding_dim, latent_dim, vocab_size=inference_decoder.dictionary_size)


optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

def loss_function(real, pred,  z, z_mean, z_log_var):
    reconstruction_loss = tf.reduce_mean(
        keras.losses.categorical_crossentropy(real, pred)
    )
    reconstruction_loss *= 32 * 32
    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss = tf.reduce_mean(kl_loss)
    kl_loss *= -0.5
    total_loss = reconstruction_loss + kl_loss

    return total_loss


"""
## Train the VAE
"""
validx = None
validy1 =  None
validy2 =  None

checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer = optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)

# start_epoch = 0
# if ckpt_manager.latest_checkpoint:
#   start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
#   # restoring the latest checkpoint in checkpoint_path
#   ckpt.restore(ckpt_manager.latest_checkpoint)

def train_step(input_image, decoder_input, decoder_target):
    loss = 0

    with tf.GradientTape() as tape:
        feature_embeddings, z, z_mean, z_log_var = encoder(input_image)
      #  print("training embedding x:", feature_embeddings.shape)
      #  print("training z:", z.shape)
        hidden = z # decoder.reset_state(batch_size=BATCH_SIZE)
      #  print("decoder target shape:", decoder_target.shape)
        decoder_frames = decoder_input[:, 0, :] #tf.expand_dims(decoder_input[:, 0, :],1)
      #  print("decoder input shape", decoder_frames.shape)
        for i in range(1, decoder_target.shape[1]):
            # passing the features through the decoder
            predictions, hidden, _ = decoder(decoder_frames, feature_embeddings, hidden)
            #print("prediction from decoder ", predictions.shape)
            loss += loss_function(decoder_target[:, i], predictions, z, z_mean, z_log_var)

            # using teacher forcing
            decoder_frames = decoder_input[:, i, :] #tf.expand_dims(decoder_input[:, i, :],1)
    total_loss = (loss / int(decoder_target.shape[1]))

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss
EPOCHS = 5
valid_x = None
for epoch in range(0, EPOCHS):
    print("Training Epoch : ", epoch)
    total_loss = 0
    for batch in range(5):
        train_x = np.empty((BATCH_SIZE, 32, 32, FEATURE_MAP_DIMENSION))
        train_y_1 = np.empty((BATCH_SIZE, MAX_WORD_LENGTH, ONE_HOT_WORD_LENGTH))
        train_y_2 = np.empty((BATCH_SIZE, MAX_WORD_LENGTH, ONE_HOT_WORD_LENGTH))
        random_idx = []
        for j in range(BATCH_SIZE):
            random_idx.append([random.randint(0, len(training_datasets) - 1), random.randint(0, BATCH_SIZE-1)])
        for idx, choice in enumerate(random_idx):
            # choice[0] is which gameplay we pick
            # choice[1] is which step in the gameplay we pick
            # Find list of IDs
            each_training_sample = training_datasets[choice[0]]
            train_x[idx, ] = each_training_sample[0][choice[1]]
            input_sequence = each_training_sample[1][choice[1]]
            output_sequence = each_training_sample[2][choice[1]]

            # convert sequence to vector
            decoder_input_data = np.zeros(
                (1, MAX_WORD_LENGTH, inference_decoder.dictionary_size),
                dtype=np.float32)
            decoder_target_data = np.zeros(
                (1, MAX_WORD_LENGTH, inference_decoder.dictionary_size),
                dtype=np.float32)

            input_sequence_list = input_sequence.split()
            output_sequence_list = output_sequence.split()
            assert (len(input_sequence_list) == len(output_sequence_list))
            for word_idx in range(len(input_sequence_list)):
                input_word = input_sequence_list[word_idx]
                output_word = output_sequence_list[word_idx]
                # if input_word == '(' or output_word == ')':
                #     print(input_word, output_word)
                # TODO : increase length of sentence
                if word_idx == MAX_WORD_LENGTH - 1:
                    break
                decoder_input_data[0][word_idx][inference_decoder.word_to_index_mapping[input_word]] = 1.
                decoder_target_data[0][word_idx][inference_decoder.word_to_index_mapping[output_word]] = 1.
            train_y_1[idx, ] = decoder_input_data
            train_y_2[idx, ] = decoder_target_data
        valid_x = train_x
        batch_loss, t_loss = train_step(train_x, train_y_1, train_y_2)
        total_loss += t_loss
        print("loss :", total_loss)
    ckpt_manager.save()

dictionary = utils.Inference(board_size=21)
def evaluate(image):
    attention_plot = np.zeros((MAX_WORD_LENGTH, latent_dim))

   # hidden = decoder.reset_state(batch_size=1)

    features, z, _, _ = encoder(image)

    dec_input = np.zeros((1, 450), dtype=np.float32)
    # Populate the first character of target sequence with the start character.
    dec_input[0, 448] = 1.
    result = []

    for i in range(MAX_WORD_LENGTH):
       # print(dec_input.shape, features.shape, hidden.shape)
        predictions, z, attention_weights = decoder(dec_input, features, z)

       # attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = np.argmax(predictions[-1, :])
        result.append(dictionary.index_to_word_mapping[predicted_id])

        if dictionary.index_to_word_mapping[predicted_id] == ')':
            return result, attention_plot

        dec_input = np.zeros((1, 450), dtype=np.float32)
        dec_input[0, predicted_id] = 1.

    #attention_plot = attention_plot[:len(result), :]
    return result, attention_plot
for i in range(10):
    result, attention_plot = evaluate(valid_x[i:i + 1, :, :, :])
    print(result)
print("done")