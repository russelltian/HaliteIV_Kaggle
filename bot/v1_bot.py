import numpy as np
import tensorflow._api.v2.compat.v1 as tf

tf.compat.v1.disable_eager_execution()

from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *


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
        saver = tf.train.import_meta_graph('model.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))

        # Get tensorflow training graph
        self.sess, self.saver = sess, saver

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
        if not self.sess or not self.saver:
            print("Error, no model found")
            return {}
        sess, saver = self.sess, self.saver
        # print([n.name for n in tf.get_default_graph().as_graph_def().node])
        frame_node = tf.get_default_graph().get_collection('frames')[0]
        loss_node = tf.get_default_graph().get_collection('loss')[0]
        my_ships_node = tf.get_collection('my_ships')[0]
        turns_left_node = tf.get_default_graph().get_collection('turns_left')[0]
        train_x = np.zeros((32, 32, 4))
        ship_info = np.zeros((32, 32))
        # add padding

        halite_map = np.array(halite_map)
        ship_map = np.array(ship_map)
        cargo_map = np.array(cargo_map)
        pad_offset = (32 - size) // 2
        print(halite_map.shape)
        moves_node = tf.get_default_graph().get_collection('m_logits')[0]
        spawn_node = tf.get_default_graph().get_collection('s_logits')[0]
        train_features = np.stack((halite_map, ship_map, cargo_map, shipyard_map), axis=-1)
        train_x[pad_offset:pad_offset + size, pad_offset:pad_offset + size, :] = train_features
        X = [train_x]
        X = np.array(X)
        ship_info[pad_offset:pad_offset + size, pad_offset:pad_offset + size] = myship_map
        my_ships = [ship_info]
        my_ships = np.expand_dims(np.array(my_ships), -1)
        turns_left = np.array(config.episodeSteps - obs.step - 1).reshape(1, 1)
        feed_dict = {frame_node: X, turns_left_node: turns_left, my_ships_node: my_ships}

        print("Training data dimension:", X.shape)
        padded_moves, spawn_or_not = sess.run([moves_node, spawn_node], feed_dict)
        print('padded_moves is', padded_moves.shape)
        padded_moves = padded_moves[0]
        ship_moves = padded_moves[pad_offset:pad_offset + size, pad_offset:pad_offset + size, :]
        valid_move = ["STAY", "EAST", "WEST", "SOUTH", "NORTH", "CONVERT"]

        print("spawn or not is", spawn_or_not)
        for ship in current_player.ships:
            position = self.convert_kaggle_2D_to_coordinate_2D(this_turn.size, list(ship.position))
            print(ship_moves[position[0]][position[1]])
            this_action = valid_move[np.argmax(ship_moves[position[0]][position[1]])]
            if this_action == "STAY":
                continue
                #actions[ship.id] = "NORTH"
            else:
                actions[ship.id] = this_action


        return actions