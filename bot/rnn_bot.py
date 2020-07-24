from kaggle_environments.envs.halite.helpers import *
import numpy as np
import tensorflow as tf
from train import utils


class LSTM_Bot(utils.Gameplay):
    """
    This class inherits the base class for agents,
    it has a set of prepared data and helper function to help the further development.
    """
    def __init__(self, obs, config):
        super().__init__(obs, config)
        self.encoder_model, self.decoder_model = self.load_model()

    def load_model(self):
        e = tf.keras.models.load_model('encoder.h5')
        d = tf.keras.models.load_model('decoder.h5')
        return e, d

    def reset_board(self, obs, config):
        super().reset_board(obs, config)

    def normalize(self):
        super().normalize()

    def agent(self, obs, config):
        """Central function for an agent.
            Relevant properties of arguments:
            obs:
                halite: a one-dimensional list of the amount of halite in each board space
                player: integer, player id, generally 0 or 1
                players: a list of players, where each is:
                    [halite, { 'shipyard_uid': position }, { 'ship_uid': [position, halite] }]
                step: which turn we are on (counting up)
            Should return a dictionary where the key is the unique identifier string of a ship/shipyard
            and action is one of "CONVERT", "SPAWN", "NORTH", "SOUTH", "EAST", "WEST"
            ("SPAWN" being only applicable to shipyards and the others only to ships).
        """
        this_turn = self
        current_player = this_turn.board.current_player
        size = self.board_size

        actions = {}

        # add padding

        pad_offset = (32 - size) // 2
        valid_move = ["STAY", "EAST", "WEST", "SOUTH", "NORTH", "CONVERT"]
        # Define sampling models
        num_encoder_tokens = 6
        latent_dim = 32

        encoder_input_data = np.zeros(
            (1, 441, 6),
            dtype='float32')

        decoder_input_data = np.ones(
            (1, 441, 6),
            dtype='float32')

        count = 0
        for row in self.my_ships_location:
            for item in row:
                # print(count)
                encoder_input_data[:, count, int(item)] = 1.
                decoder_input_data[:, count, 5] = 1.
                count += 1
        encoder_model = self.encoder_model
        decoder_model = self.decoder_model
        num_decoder_tokens = 6

        def decode_sequence(input_seq):
            # Encode the input as state vectors.
            states_value = encoder_model.predict(input_seq)

            # Generate empty target sequence of length 1.
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            # Populate the first character of target sequence with the start character.
            target_seq[0, 0, 0] = 1.

            # Sampling loop for a batch of sequences
            # (to simplify, here we assume a batch of size 1).
            stop_condition = False
            decoded_sentence = ''
            while not stop_condition:
                output_tokens, h, c = decoder_model.predict(
                    [target_seq] + states_value)

                # Sample a token
                sampled_token_index = np.argmax(output_tokens[0, -1, :])
                sampled_char = str(sampled_token_index)
                decoded_sentence += sampled_char

                # Exit condition: either hit max length
                # or find stop character.
                if (sampled_char == '\n' or
                        len(decoded_sentence) > 441):
                    stop_condition = True

                # Update the target sequence (of length 1).
                target_seq = np.zeros((1, 1, num_decoder_tokens))
                target_seq[0, 0, sampled_token_index] = 1.

                # Update states
                states_value = [h, c]

            return decoded_sentence

        input_seq = encoder_input_data
        result = decode_sequence(input_seq)
        for ship in current_player.ships:
            position = self.convert_kaggle2D_to_kaggle1D(this_turn.board_size, list(ship.position))
            print(valid_move[np.argmax(result[position])])
        return actions
