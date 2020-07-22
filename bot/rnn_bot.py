from kaggle_environments.envs.halite.helpers import *
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import LSTM, Dense


class Gameplay(object):
    def __init__(self, obs, config):
        size = config.size
        self.size = size
        self.ships = np.zeros((size, size))
        self.shipyards = np.zeros((size, size))
        self.board = Board(obs, config)
        self.me = self.board.current_player.id
        self.sess = None
        self.saver = None
        self.encoder_model,self.decoder_model = self.load_model()
    def print_board(self):
        board = self.board
        print(board)
    def reset_board(self, obs, config):
        # The size of the board must be constant
        assert(self.size == config.size)
        size = self.size
        self.ships = np.zeros((size, size))
        self.shipyards = np.zeros((size, size))
        self.board = Board(obs, config)
        # The new observation must still serve for the same player
        assert(self.me == self.board.current_player.id)
    # Converts position from 1D to 2D representation in (left most col, left most row)
    def get_col_row(self, size: int, pos: int):
        return pos % size, pos // size
    def load_model(self):
        e = tf.keras.models.load_model('encoder.h5')
        d = tf.keras.models.load_model('decoder.h5')
        return e, d
    # convert top left coordinate to (left row, left col)
    def get_2D_col_row(self, size: int, pos: int):
        top_left_row = pos // size
        top_left_col = pos % size
        return top_left_row, top_left_col

    def test_get_2D_col_row(self, size=21):
        assert (self.get_2D_col_row(size, 0) == (0, 0))
        assert (self.get_2D_col_row(size, 10) == (0, 10))
        assert (self.get_2D_col_row(size, 413) == (19, 14))
        assert (self.get_2D_col_row(size, 440) == (20, 20))

    def get_to_pos(self, size: int, pos: int, direction: str):
        col, row = self.get_col_row(size, pos)
        if direction == "NORTH":
            return pos - size if pos >= size else size ** 2 - size + col
        elif direction == "SOUTH":
            return col if pos + size >= size ** 2 else pos + size
        elif direction == "EAST":
            return pos + 1 if col < size - 1 else row * size
        elif direction == "WEST":
            return pos - 1 if col > 0 else (row + 1) * size - 1

    def convert_kaggle_2D_to_coordinate_2D(self, size: int, pos: List[int]):
        """
        Convert the target position from coordinate with bottom left point as origin to
        coordinate where the top left point is the origin
        :param size:
        :param pos:
        :return:
        """
        assert (len(pos) == 2)
        row = size - 1 - pos[1]
        col = pos[0]
        assert (0 <= row < size)
        assert (0 <= col < size)
        return row, col

    def convert_kaggle_2D_to_coordinate_1D(self, size: int, pos: List[int]):
        """
        Convert the target position from coordinate with bottom left point as origin to
        coordinate where the top left point is the origin
        :param size:
        :param pos:
        :return:
        """
        assert (len(pos) == 2)
        row = size - 1 - pos[1]
        col = pos[0]
        assert (0 <= row < size)
        assert (0 <= col < size)
        return row*size + col


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
        # this_turn.print_board()
        current_player = this_turn.board.current_player
        size = self.size
        opponents = this_turn.board.opponents
        halite_map = np.zeros((size, size))
        ship_map = np.zeros((size, size))
        cargo_map = np.zeros((size, size))
        shipyard_map = np.zeros((size,size))
        myship_map = np.zeros((size,size))

        # Load halite
        for i in range(size):
            for j in range(size):
                halite_map[i][j] = obs.halite[i * size + j]

        # print(halite_map)
        # Load current player
        for ship in current_player.ships:
            position = self.convert_kaggle_2D_to_coordinate_2D(this_turn.size, list(ship.position))
            ship_map[position[0]][position[1]] = 2
            cargo_map[position[0]][position[1]] = ship.halite
            myship_map[position[0]][position[1]] = 1
        for shipyard in current_player.shipyards:
            position = self.convert_kaggle_2D_to_coordinate_2D(this_turn.size, list(shipyard.position))
            shipyard_map[position[0]][position[1]] = 2

        for opponent in opponents:
            for ship in opponent.ships:
                position = self.convert_kaggle_2D_to_coordinate_2D(this_turn.size, list(ship.position))
                ship_map[position[0]][position[1]] = 1
                cargo_map[position[0]][position[1]] = ship.halite
            for shipyard in opponent.shipyards:
                position = self.convert_kaggle_2D_to_coordinate_2D(this_turn.size, list(shipyard.position))
                shipyard_map[position[0]][position[1]] = 1

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
        for row in ship_map:
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
            position = self.convert_kaggle_2D_to_coordinate_1D(this_turn.size, list(ship.position))
            print(valid_move[np.argmax(result[position])])
        return actions