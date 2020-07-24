from kaggle_environments.envs.halite.helpers import *
import numpy as np
import tensorflow as tf
from train import utils

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
        self.vae = self.load_model()
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
        vae = tf.keras.models.load_model('vae.h5')
        return vae
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
        size = 21
        pad_offset = 6

        input_image = np.zeros(
            (1, 32, 32, 3),
            dtype='float32')

        # Load current player
        for ship in current_player.ships:
            position = self.convert_kaggle_2D_to_coordinate_2D(this_turn.size, list(ship.position))
            input_image[0][position[0] + pad_offset][position[1] + pad_offset][0] = 1.0

        # Load halite
        for i in range(size):
            for j in range(size):
                input_image[0][i+pad_offset][j+pad_offset][1] = obs.halite[i * size + j]

        # print(halite_map)
        for shipyard in current_player.shipyards:
            position = self.convert_kaggle_2D_to_coordinate_2D(this_turn.size, list(shipyard.position))
            input_image[0][position[0]+pad_offset][position[1]+pad_offset][2] = 1.0

        actions = {}
        valid_move = ["STAY", "EAST", "WEST", "SOUTH", "NORTH", "CONVERT"]
        # Define sampling models

        vae = self.vae

        result = vae.predict(input_image)

        print("result is", result)
        print("result size", result.shape)

        for ship in current_player.ships:
            position = self.convert_kaggle_2D_to_coordinate_1D(this_turn.size, list(ship.position))
            print(valid_move[np.argmax(result[position])])
        return actions


class VaeBot(utils.Gameplay):
    """
    This bot inherits the base class for agent bot,
    it comes with a set of preprocessed data as well as helper function.
    For all the new data you added, you can reset them in the reset function,
    and scale them in the normalize function
    """
    def __init__(self, obs, config):
        super().__init__(obs, config)
        self.vae = self.load_model()

    def reset_board(self, obs, config):
        super().reset_board(obs, config)

    def normalize(self):
        super().normalize()

    def load_model(self):
        vae = tf.keras.models.load_model('vae.h5')
        return vae

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
        size = 21
        pad_offset = 6

        input_image = np.zeros(
            (1, 32, 32, 3),
            dtype='float32')

        # Load current player
        for ship in current_player.ships:
            position = self.convert_kaggle2D_to_upperleft2D(this_turn.board_size, list(ship.position))
            input_image[0][position[0] + pad_offset][position[1] + pad_offset][0] = 1.0

        # Load halite
        for i in range(size):
            for j in range(size):
                input_image[0][i+pad_offset][j+pad_offset][1] = obs.halite[i * size + j]

        # print(halite_map)
        for shipyard in current_player.shipyards:
            position = self.convert_kaggle2D_to_upperleft2D(this_turn.board_size, list(shipyard.position))
            input_image[0][position[0]+pad_offset][position[1]+pad_offset][2] = 1.0

        actions = {}
        valid_move = ["STAY", "EAST", "WEST", "SOUTH", "NORTH", "CONVERT"]
        # Define sampling models

        vae = self.vae

        result = vae.predict(input_image)

        print("result is", result)
        print("result size", result.shape)

        for ship in current_player.ships:
            position = self.convert_kaggle2D_to_kaggle1D(this_turn.board_size, list(ship.position))
            print(valid_move[np.argmax(result[position])])
        return actions