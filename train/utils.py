import json
import os
import sys

import numpy as np

sys.path.append("../")
from train import geometry


class Halite(object):
    """

    """

    def __init__(self):

        self.replay = None
        # halite per cell
        self.halite = []
        self.ship_actions = []
        self.shipyard_actions = []
        self.ship_position = []
        self.cargo = []
        self.turns_left = []

    def load_replay(self, path: str):
        """
                load replay json file from halite website
                :return:
                """
        # check there is a file in the given path
        assert (os.path.isfile(path))
        with open(path) as f:
            self.replay = json.loads(f.read())

    def load_data(self):
        """
        After replay is loaded, fetch raw data for training
        :return:
        """
        assert (self.replay is not None)
        # parameters that is used to retrieve specific piece of data
        game_config = self.replay["configuration"]
        number_of_players = len(self.replay["rewards"])
        map_size = game_config["size"]
        self.halite = self.load_halite(map_size)
        self.ship_actions, self.shipyard_actions = self.load_moves(map_size, number_of_players)
        self.ship_position, self.cargo = self.load_ship_position(map_size)
        total_step = self.cargo.shape[0]
        self.turns_left = np.array([total_step - i for i in range(total_step)])

    def load_halite(self, map_size: int):
        """
        Load the amount of halite on the map at each turn
        :param map_size: the size of the map, assume the map is a square
        :return: a 3D numpy array with dimension of steps - 1 x map_size x map_size
        """
        halite = []
        assert (self.replay is not None)
        for step, content in enumerate(self.replay["steps"]):
            # Get board observation
            if step == 0:
                continue
            observation = self.replay["steps"][step - 1][0]["observation"]

            # load the amount of halite on the map
            raw_energy_grid = observation["halite"]
            one_step = []
            assert (len(raw_energy_grid) == map_size ** 2)
            for i in range(map_size):
                one_row = []
                for j in range(map_size):
                    one_row.append(raw_energy_grid[i * map_size + j])
                one_step.append(one_row)
            halite.append(one_step)
        halite = np.array(halite)
        total_step = len(self.replay["steps"])
        assert (halite.shape[0] == total_step - 1)
        assert (halite.shape[1] == map_size)
        assert (halite.shape[2] == map_size)
        halite = halite/1000  # round it
        return halite

    def load_moves(self, map_size: int, num_of_players: int):
        """
        TODO: Get to work on players
        Loads the ship actions and shipyard actions for player 0 at each turn
        :return: a numpy 3D array
        """
        valid_actions = ["STAY", "EAST", "WEST", "SOUTH", "NORTH", "CONVERT"]
        ships_action = []
        shipyards_action = []
        # Iterate through each step of the game to get step based information
        for step, content in enumerate(self.replay["steps"]):
            if step == 0:
                continue
            # Get board observation
            # observation = content[0]["observation"]
            observation = self.replay["steps"][step - 1][0]["observation"]
            # Declare processing information
            # ship action and shipyard action per step
            step_ships_action = np.zeros((map_size, map_size), np.int32)
            step_shipyard_action = np.zeros((map_size, map_size), np.int32)

            # Load ship moves for all active players
            for pid in range(len(content)):
                if "player" not in content[pid]["observation"]:
                    continue
                player_id = content[pid]["observation"]["player"]

                # load player 0's information
                # TODO: change it to all players
                if player_id == 0:
                    player_observation = observation["players"][player_id]
                    # Get halite, shipyard, ship information of the player
                    player_shipyard = player_observation[1]
                    player_ship = player_observation[2]
                    # load action
                    for ship_shipyard_id, move in content[pid]["action"].items():
                        if move == "SPAWN":
                            # check it is a shipyard
                            assert (ship_shipyard_id in player_shipyard)
                            # Get shipyard location
                            shipyard_pos_1d = player_shipyard[ship_shipyard_id]
                            shipyard_pos_2d = geometry.get_2D_col_row(map_size, shipyard_pos_1d)
                            step_shipyard_action[shipyard_pos_2d[0]][shipyard_pos_2d[1]] = 1
                        else:
                            # check it is a valid move for ships
                            assert (move in valid_actions)
                            # get information of the ship
                            ship_info = player_ship[ship_shipyard_id]
                            assert (len(ship_info) == 2)  # [pos,cargo]
                            ship_pos_1d = ship_info[0]
                            ship_pos_2d = geometry.get_2D_col_row(map_size, ship_pos_1d)
                            step_ships_action[ship_pos_2d[0]][ship_pos_2d[1]] = valid_actions.index(move)
                    # print(step_ships_action)
                    # print(step_shipyard_action)
                    ships_action.append(step_ships_action)
                    shipyards_action.append(step_shipyard_action)
            # Load training features

        ships_action = np.array(ships_action)
        shipyards_action = np.array(shipyards_action)
        return ships_action, shipyards_action

    def load_ship_position(self, map_size):
        """
        TODO: Include 2d shipyard positions
        Loads all active ships 2D positions for player 0 at each turn
        :return:
        """
        valid_actions = ["EAST", "WEST", "SOUTH", "NORTH", "CONVERT"]
        ships_position = []
        ships_cargo = []
        # Iterate through each step of the game to get step based information
        for step, content in enumerate(self.replay["steps"]):
            if step == 0:
                continue
            # Get board observation
            # observation = content[0]["observation"]
            observation = self.replay["steps"][step - 1][0]["observation"]

            step_ships_position = np.zeros((map_size, map_size), np.int32)
            # step_shipyard_action = np.zeros((map_size, map_size))
            step_ship_cargo = np.zeros((map_size, map_size), np.float32)
            # Load ship moves for all active players
            for pid in range(len(content)):
                if "player" not in content[pid]["observation"]:
                    continue
                player_id = content[pid]["observation"]["player"]

                # load player 0's information
                # TODO: change it to all players
                if player_id == 0:
                    player_observation = observation["players"][player_id]
                    # Get halite, shipyard, ship information of the player
                    # player_shipyard = player_observation[1]
                    player_ship = player_observation[2]
                    # load action
                    for ship_id, ship_info in player_ship.items():
                        assert (len(ship_info) == 2)  # ship_info : [pos,cargo]
                        ship_pos_1d = ship_info[0]
                        ship_pos_2d = geometry.get_2D_col_row(map_size, ship_pos_1d)
                        step_ships_position[ship_pos_2d[0]][ship_pos_2d[1]] = 1
                        step_ship_cargo[ship_pos_2d[0]][ship_pos_2d[1]] = ship_info[1]
                    ships_position.append(step_ships_position)
                    ships_cargo.append(step_ship_cargo)
        ships_position = np.array(ships_position)
        ships_cargo = np.array(ships_cargo)
        #print(ships_position.shape)
        return ships_position, ships_cargo

    def get_training_data(self, dim=32):
        """
        After loading the raw data, do further processing to get the data prepared for training
        :param dim: the training data has the shape of [ None, dim, dim]
        :return:
        """
        assert (self.replay is not None and self.halite is not None)
        #print(self.halite.shape)
        # print(self.ship_actions.shape)
        assert (self.halite.shape[0] == self.ship_actions.shape[0])
        step = self.halite.shape[0]
        shape = self.halite.shape[1]
        train_x = np.zeros((step, dim, dim, 3))
        train_y = np.zeros((step, dim, dim))
        train_feature_mix = np.stack([self.halite, self.ship_position, self.cargo], axis=-1)
        # add padding
        if shape != dim:
            pad_offset = (dim - self.halite.shape[1]) // 2
            train_x[:, pad_offset:pad_offset + shape, pad_offset:pad_offset + shape, :] = train_feature_mix
            train_y[:, pad_offset:pad_offset + shape, pad_offset:pad_offset + shape] = self.ship_actions
            return train_x, train_y
        return train_feature_mix, self.ship_actions.copy()

    def get_my_ships(self, dim=32):
        """
                After loading the raw data, get current player ship positions with padding
                :param dim: the training data has the shape of [ None, dim, dim]
                :return:
                """
        assert (self.replay is not None and self.ship_position is not None)
        step = self.ship_position.shape[0]
        shape = self.ship_position.shape[1]
        ship_info = np.zeros((step, dim, dim))
        # add padding
        if shape != dim:
            pad_offset = (dim - self.halite.shape[1]) // 2
            ship_info[:, pad_offset:pad_offset + shape, pad_offset:pad_offset + shape] = self.ship_position
            return ship_info
        return self.ship_position.copy()


# game = Halite()
# game.load_replay("1208740.json")
# #game.load_halite(21)
# game.load_data()
# game.get_training_data()