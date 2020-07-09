import numpy as np
import tensorflow._api.v2.compat.v1 as tf

tf.compat.v1.disable_eager_execution()

from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *


class Gameplay(object):
    def __init__(self, obs, config):
        self.obs = obs
        self.config = config
        size = config.size
        self.size = size
        self.ships = np.zeros((size, size))
        self.shipyards = np.zeros((size, size))
        self.board = Board(obs, config)
        self.me = self.board.current_player
        self.model = self.load_model()

    def print_board(self):
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

    def get_training_data(self, dim=32):
        return

    def load_model(self):
        sess = tf.Session()
        saver = tf.train.import_meta_graph('model_9.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))

        # Get tensorflow training graph
        return sess, saver

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
        # this_turn.print_board()
        current_player = this_turn.board.current_player
        size = self.size
        halite_map = np.zeros((size, size))
        ship_map = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                halite_map[i][j] = this_turn.obs.halite[i * size + j]

        # print(halite_map)
        for ship in current_player.ships:
            position = self.convert_kaggle_2D_to_coordinate_2D(this_turn.size, list(ship.position))
            ship_map[position[0]][position[1]] = 1
        actions = {}
        sess, saver = self.load_model()
        # print([n.name for n in tf.get_default_graph().as_graph_def().node])
        frame_node = tf.get_default_graph().get_collection('frames')[0]
        loss_node = tf.get_default_graph().get_collection('loss')[0]
        train_x = np.zeros((32, 32, 2))

        # add padding

        halite_map = np.array(halite_map)
        ship_map = np.array(ship_map)
        pad_offset = (32 - size) // 2
        print(halite_map.shape)
        moves_node = tf.get_default_graph().get_collection('m_logits')[0]
        train_features = np.stack((halite_map, ship_map), axis=-1)
        train_x[pad_offset:pad_offset + size, pad_offset:pad_offset + size, :] = train_features
        X = [train_x]
        X = np.array(X)

        feed_dict = {frame_node: X}
        print("Training data dimension:", X.shape)
        padded_moves = sess.run([moves_node], feed_dict)
        padded_moves = padded_moves[0][0]
        ship_moves = padded_moves[pad_offset:pad_offset + size, pad_offset:pad_offset + size, :]
        valid_move = ["STAY", "EAST", "WEST", "SOUTH", "NORTH", "CONVERT"]
        for ship in current_player.ships:
            position = self.convert_kaggle_2D_to_coordinate_2D(this_turn.size, list(ship.position))
            print(ship_moves[position[0]][position[1]])
            this_action = valid_move[np.argmax(ship_moves[position[0]][position[1]])]
            if this_action == "STAY":
                actions[ship.id] = "EAST"
            else:
                actions[ship.id] = this_action
        return actions