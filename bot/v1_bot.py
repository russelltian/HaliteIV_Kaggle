import numpy as np
import tensorflow as tf

from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *

class Gameplay(object):
    def __init__(self, obs, config):
        self.obs = obs
        self.config = config
        size = config.size
        self.ships = np.zeros((size, size))
        self.shipyards = np.zeros((size, size))
        self.board = Board(obs, config)
        self.me = self.board.current_player
        
    def load_data(self):
        board = self.board
        print(board)
    # Converts position from 1D to 2D representation in (left most col, left most row)
    def get_col_row(self, size: int, pos: int):
        return pos % size, pos // size

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
        this_turn = Gameplay(obs, config)
        this_turn.load_data()
        actions = {}
        return actions

