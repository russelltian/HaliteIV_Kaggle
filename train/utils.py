import json
import os
from kaggle_environments.envs.halite.helpers import *
import sys

import numpy as np
import tensorflow as tf

sys.path.append("../")
sys.path.append("../bot/")
'''
######################## Gameplay Part ################################################################
'''
class Gameplay(object):
    """
    This is the base class that supports the agent in real game environment.
    It takes the same input as the agent does in real game,
    and provisions data processing utilities to convert the data to the format
    that will be fed into our later ML based agent.
    Further optimization and rework can be done on the derived class.
    """

    def __init__(self, obs, config):
        self.board = Board(obs, config)
        self.obs = obs
        size = config["size"]
        self.board_size = size  # Dimension of the SQUARE board
        self.current_player_id = self.board.current_player_id  # your player id in this game
        # self.halite_map = np.zeros((size, size), np.float32)  # 2D map of halite
        # self.my_ships_location = np.zeros((size, size), np.float32)  # 2D map of your ships location
        self.my_cargo = np.zeros((size, size), np.float32)  # 2D map of your ships cargo
        # self.my_shipyards_location = np.zeros((size, size), np.float32)  # 2D map of your shipyards location
        # # opponents ship locations
        # self.opponent_ships_location = np.zeros((size, size), np.float32)


        # Store information (Note, if this flow got updated, also update reset board function)
        # Load ship location, shipyard location, and ship cargo in 2D matrix
        # Load enemies ship location
        self.get_ships_information()
        # self.get_halite()  # Load halite in 2D matrix
        # self.normalize()  # Normalization is defined here



    def get_ships_information(self):
        """
        Get current player's ship location, shipyard location, and cargo on the ship
        :return:
        """
        # this_turn.print_board()
        board = self.board
        size = self.board_size
        current_player = board.current_player
        other_players = board.opponents
        for ship in current_player.ships:
            position = self.convert_kaggle2D_to_upperleft2D(size, list(ship.position))
            # self.my_ships_location[position[0]][position[1]] = 1.
            self.my_cargo[position[0]][position[1]] = ship.halite
        # for shipyard in current_player.shipyards:
        #     position = self.convert_kaggle2D_to_upperleft2D(size, list(shipyard.position))
        #     self.my_shipyards_location[position[0]][position[1]] = 1.
        #
        # for player in other_players:
        #     for ship in player.ships:
        #         position = self.convert_kaggle2D_to_upperleft2D(size, list(ship.position))
                # self.opponent_ships_location[position[0]][position[1]] = 1.
                # assert(self.my_ships_location[position[0]][position[1]] != 1.)

    # def get_halite(self):
    #     """
    #     Get halite
    #     :return:
    #     """
    #     size = self.board_size
    #     for i in range(size):
    #         for j in range(size):
    #             self.halite_map[i][j] = self.obs['halite'][i * size + j]
    #
    # def normalize(self):
    #     """
    #     Normalize based on the need
    #     :return:
    #     """
    #     self.halite_map = self.halite_map / 100
    #     self.my_cargo = self.my_cargo / 100

    # def pad(self, input_2D_matrix):
    #     pass

    def print_board(self):
        print(self.board)

    def reset_board(self, obs, config):
        """
        Reset board information
        :param obs:
        :param config:
        :return:
        """
        # The size of the board must be constant
        assert (self.board_size == config["size"])
        self.board = Board(obs, config)
        self.obs = obs
        # The new observation must still serve the same player
        assert (self.current_player_id == self.board.current_player_id)

        # # Store information
        self.get_ships_information()
        # self.get_halite()
        # self.normalize()

    def convert_kaggle1D_to_upperleft2D(self, size: int, pos: int):
        """
         convert 1D position starting from top left of matrix to (row, col) originated at top left of matrix
        :param size:
        :param pos:
        :return:
        """
        top_left_row = pos // size
        top_left_col = pos % size
        return top_left_row, top_left_col

    def convert_kaggle2D_to_upperleft2D(self, size: int, pos: List[int]):
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

    def convert_kaggle2D_to_kaggle1D(self, size: int, pos: List[int]):
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
        return row * size + col

    def find_actions_in_ship_proximity(self, size: int, actions, ship_pos: int):
        """
        Find the first (if any) action that is within 1 space's proximity to the ship's position.
        If no such action exists, return None
        :param actions:
        :param ship_pos:
        :return:
        """
        # TODO: improve robustness in terms of considering crossing the board boundary
        assert (type(ship_pos) == int)
        horizontal_proximity = [ship_pos - 1, ship_pos + 1, ship_pos + size, ship_pos - size]
        print("horizontal proximity", horizontal_proximity)
        diagonal_proximity = [ship_pos - size - 1, ship_pos - size + 1, ship_pos + size - 1, ship_pos + size + 1]
        print("diagonal proximity", diagonal_proximity)
        for pos in actions.keys():
            if pos in horizontal_proximity and actions[pos] != 'NO':
                return actions[pos]
        for pos in actions.keys():
            if pos in diagonal_proximity and actions[pos] != 'NO':
                return actions[pos]
        return None



'''
######################## Training Part ################################################################
'''
from bot import vae_bot


class HaliteV2(object):
    """
    This is the wrapper class for preparing training data for ML training mode with Json files.
    It reads the Json file and convert the information to useful meta data and a list of
    "Gameplay" object that represents single step of the game, all the core data processing is done in Gameplay,
    and this class will fetch and reformat the data within "Gameplay" to make it training ready.
    """
    # TODO: currently ignore the winner of the game, just train based on the first player cuz they all good for now
    def __init__(self, path: str, ignore_winner=True):
        self.replay, self.config = self.load_replay(path)
        self.total_turns = self.find_total_turns()
        self.winner_id = self.find_winner() if not ignore_winner else 0
        self.game_play_list = self.build_game_play_list()

        self.turns_left = []
        self.spawn = []

        # my ship action is only available in supervise learning training mode
        self.valid_move = ["STAY", "EAST", "WEST", "SOUTH", "NORTH", "CONVERT"]
        #self.my_ship_action_list_2D, self.my_shipyard_action_list_2D, self.move_sequence = self.build_my_ship_action_list()
        self.move_sequence = self.build_my_ship_action_list()
        # For training
        # For training
        self.ship_position = None
        self.ship_actions = None
        self.halite = None
        self.cargo = None
        self.shipyard_position = None
        self.opponent_ship_position = None
        self.meta_data = None
    def load_replay(self, path: str):
        """
                load replay json file from halite website
                :return:
                """
        # check there is a file in the given path
        assert (os.path.isfile(path))
        with open(path) as f:
            replay = json.loads(f.read())
        # Load configuration
        config = replay["configuration"]
        return replay, config

    def find_total_turns(self):
        """
        Find how many turns this game has, note the game terminates when there is only one player
        left, so the game does not have to last for full turns
        :return:
        """
        assert (self.config is not None)
        episode_steps = self.config["episodeSteps"]
        actual_steps = len(self.replay["steps"])
        assert (actual_steps <= episode_steps)
        return actual_steps

    def find_winner(self):
        """
        Find the id of the player who gets most reward at the last of the game
        :return:
        """
        assert (self.replay is not None)
        assert (self.replay["steps"][-1] is not None)
        player_reward = []
        last_step = self.replay["steps"][- 1]
        first_step = self.replay["steps"][1]
        # has to be 4 player assume
        assert (len(last_step) == len(first_step))
        for player in last_step:
            player_reward.append(player["reward"])
        return player_reward.index(max(player_reward))

    def convert_to_game_play(self, step: int):
        """
        Convert a step of replay from json file to the format that supports by halite helper function
        and can be used directly in the real time game play
        :return:
        """
        assert (self.replay is not None)
        assert (self.config is not None)
        assert (0 < step <= self.total_turns - 1)
        observation = self.replay["steps"][step - 1][0]["observation"]
        board = Board(observation, self.config)
        assert (board.current_player_id == 0)
        #return Gameplay(observation, self.config)
        return vae_bot.VaeBot(observation, self.config)

    def build_game_play_list(self):
        """
        Convert every steps of replay to a list of "Gameplay" object that supports by kaggle helper function
        :return:
        """
        assert (self.replay is not None)
        assert (self.config is not None)
        game_play_list = []
        for step, _ in enumerate(self.replay["steps"]):
            if step == 0:
                continue
            game_play_list.append(self.convert_to_game_play(step))
        assert (len(game_play_list) == self.total_turns - 1)
        return game_play_list

    def build_my_ship_action_list(self):
        """
        Convert the ship actions to a sequence of string for seq-seq learning
        :return:
        """
        assert (self.replay is not None)
        assert (self.config is not None)
        map_size = self.config["size"]
        ships_action = []
        shipyards_action = []
        moves_sequence = []
        # convert top left coordinate to (left row, left col)
        def get_2D_col_row(size: int, pos: int):
            top_left_row = pos // size
            top_left_col = pos % size
            return top_left_row, top_left_col

        # Iterate through each step of the game to get step based information
        for step, board_info_per_step in enumerate(self.replay["steps"]):
            if step == 0:
                continue
            # Get board observation
            # observation = content[0]["observation"]
            observation = self.replay["steps"][step - 1][0]["observation"]
            # Declare processing information
            # ship action and shipyard action per step
            # step_ships_action = np.zeros((map_size, map_size), np.int32)
            # step_shipyard_action = np.zeros((map_size, map_size), np.int32)
            # ZERO if no ship yard spawned new ships

            # Load ship moves for all active players
            for pid in range(len(board_info_per_step)):
                if "player" not in board_info_per_step[pid]["observation"]:
                    continue
                player_id = board_info_per_step[pid]["observation"]["player"]

                # load player 0's information
                # TODO: change it to all players
                if player_id == self.winner_id:
                    player_observation = observation["players"][player_id]
                    # Get halite, shipyard, ship information of the player
                    player_shipyard = player_observation[1]
                    player_ship = player_observation[2]

                    # load action for 2D part
                    # for ship_shipyard_id, move in board_info_per_step[pid]["action"].items():
                    #     if move == "SPAWN":
                    #         # check it is a shipyard
                    #         assert (ship_shipyard_id in player_shipyard)
                    #         # Get shipyard location
                    #         shipyard_pos_1d = player_shipyard[ship_shipyard_id]
                    #         shipyard_pos_2d = get_2D_col_row(map_size, shipyard_pos_1d)
                    #         step_shipyard_action[shipyard_pos_2d[0]][shipyard_pos_2d[1]] = 1
                    #         # update spawn to ONE since new ship spawned
                    #     else:
                    #         # check it is a valid move for ships
                    #         assert (move in self.valid_move)
                    #         # get information of the ship
                    #         assert(ship_shipyard_id in player_ship)
                    #         ship_info = player_ship[ship_shipyard_id]
                    #         assert (len(ship_info) == 2)  # [pos,cargo]
                    #         ship_pos_1d = ship_info[0]
                    #         ship_pos_2d = get_2D_col_row(map_size, ship_pos_1d)
                    #         step_ships_action[ship_pos_2d[0]][ship_pos_2d[1]] = self.valid_move.index(move)

                    # Store all ship and shipyards action to a sequence of string
                    sequence = ""
                    shipyard_id_position = []
                    ship_id_position = []
                    ship_shipyard_move = board_info_per_step[pid]["action"]
                    for each_shipyard_id, each_shipyard_pos in player_shipyard.items():
                        assert(0 <= each_shipyard_pos < map_size**2)
                        shipyard_id_position.append([each_shipyard_id, each_shipyard_pos])
                    # sort shipyard based on location
                    shipyard_id_position = sorted(shipyard_id_position, key=lambda x: x[1])
                    for each_shipyard in shipyard_id_position:
                        # id, position
                        sequence += str(each_shipyard[1])
                        if each_shipyard[0] in ship_shipyard_move:
                            assert (ship_shipyard_move[each_shipyard[0]] == "SPAWN")
                            sequence += " SPAWN "
                        else:
                            sequence += " NO "

                    for each_ship_id, each_ship_info in player_ship.items():
                        assert(0 <= each_ship_info[0] < map_size**2)
                        ship_id_position.append([each_ship_id, each_ship_info[0]])
                    # sort ship based on location
                    ship_id_position = sorted(ship_id_position, key=lambda x: x[1])
                    for each_ship in ship_id_position:
                        # id, position
                        sequence += str(each_ship[1])
                        if each_ship[0] in ship_shipyard_move:
                            sequence += " " + str(ship_shipyard_move[each_ship[0]]) + " "
                        else:
                            sequence += " NO "
                    # print(sequence)
                    # print(step_ships_action)
                    # print(step_shipyard_action)
                    moves_sequence.append(sequence)
                    # ships_action.append(step_ships_action)
                    # shipyards_action.append(step_shipyard_action)
            # Load training features

        # ships_action = np.array(ships_action)
        # shipyards_action = np.array(shipyards_action)
        # return ships_action, shipyards_action, moves_sequence
        return moves_sequence

    # def build_training_data_for_lstm(self):
    #     """
    #     The shape is (399, 441, 6) for now
    #     :return:
    #     """
    #     X = []
    #     Y = []
    #     assert (len(self.game_play_list) == self.total_turns - 1)
    #     assert (len(self.my_ship_action_list_2D) == len(self.my_shipyard_action_list_2D) == len(self.game_play_list))
    #     for each_step in self.game_play_list:
    #         X.append(each_step.my_ships_location)
    #     for each_move in self.my_ship_action_list_2D:
    #         Y.append(each_move)
    #     X = np.array(X)
    #     Y = np.array(Y)
    #     return X, Y

    # def prepare_data_for_vae(self):
    #     """
    #     The shape is (399, 441, 6) for now
    #     :return:
    #     """
    #     ship = []
    #     halite = []
    #     shipyard = []
    #     ship_move = []
    #     cargo = []
    #     opponent_ship = []
    #     assert (len(self.game_play_list) == self.total_turns - 1)
    #     assert (len(self.my_ship_action_list_2D) == len(self.my_shipyard_action_list_2D) == len(self.game_play_list))
    #     for each_step in self.game_play_list:
    #         ship.append(each_step.my_ships_location)
    #         halite.append(each_step.halite_map)
    #         shipyard.append(each_step.my_shipyards_location)
    #         cargo.append(each_step.my_cargo)
    #         opponent_ship.append(each_step.opponent_ships_location)
    #     for each_move in self.my_ship_action_list_2D:
    #         ship_move.append(each_move)
    #     self.ship_position = np.array(ship)
    #     self.ship_actions = np.array(ship_move)
    #     self.halite = np.array(halite)
    #     self.cargo = np.array(cargo)
    #     self.shipyard_position = np.array(shipyard)
    #     self.opponent_ship_position = np.array(opponent_ship)

    def prepare_vae_encoder_input(self):
        ship_move = self.move_sequence
        input_image = []
        meta_data = []
        assert (len(self.game_play_list) == self.total_turns - 1)
        for each_step in self.game_play_list:
            input_image.append(each_step.vae_encoder_input_image)
            meta_data.append(each_step.vae_meta_data)
        input_image.append(self.game_play_list[0].vae_encoder_input_image)
        meta_data.append(self.game_play_list[0].vae_meta_data)
        first_move = self.move_sequence[0]
        ship_move.append(first_move)
        assert(len(input_image) == len(ship_move) == 400)
        return np.array(input_image), np.array(meta_data), ship_move


class Inference(object):
    def __init__(self, board_size: int):
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

        self.word_to_index_mapping = vocab_dict
        self.index_to_word_mapping = num_dict
        self.dictionary_size = len(self.index_to_word_mapping)
        assert(len(self.index_to_word_mapping) == len(self.word_to_index_mapping))
        # keras tokenizer
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=len(self.index_to_word_mapping),
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
        # tokenizer.fit_on_texts(train_captions)

    def attention_decode_sequence(self, model, input_image):
        hidden = tf.zeros((1, 512))
        features = model.encoder(input_image)
        dec_input = [[448]]

        stop_condition = False
        decoded_sentence = '( '
        decoded_word_length = 0
        decoded_actions = {}
        decoded_location = ''
        while not stop_condition:
            predictions, hidden, _ = model.decoder([dec_input, features, hidden])
            #print("predictions is", predictions)
            #print("predictions length", len(predictions[0]))
            #print("prediction is", predictions[0])
            sampled_token_index = tf.random.categorical(predictions, 1)[0][0].numpy()
            #sampled_token_index = np.argmax(predictions[0])
            print("sampled token is", sampled_token_index)
            sampled_char = self.index_to_word_mapping[int(sampled_token_index)]
            decoded_sentence += sampled_char
            decoded_sentence += " "
            decoded_word_length += 1
            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == ')' or
                    decoded_word_length > 49):
                stop_condition = True
            elif sampled_char.isdigit():
                decoded_location = sampled_char
            elif decoded_location != '':
                decoded_actions[int(decoded_location)] = sampled_char
            dec_input[0][0] = int(sampled_token_index)
            print("dec_input is", dec_input[0][0])
        print("decoded sentence ", decoded_sentence)
        return decoded_actions

    def decode_sequence(self, model, input_seq, max_sequence_length):
        # Encode the input as state vectors.
        z_mean, z_log_var, z = model.encoder(input_seq)
        states_value = z
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.dictionary_size), dtype=np.float32)
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, 448] = 1.

        stop_condition = False
        decoded_sentence = '( '
        decoded_word_length = 0
        decoded_actions = {}
        decoded_location = ''
        while not stop_condition:
            output_tokens, h = model.decoder(
                [target_seq, states_value])

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.index_to_word_mapping[int(sampled_token_index)]
            decoded_sentence += sampled_char
            decoded_sentence += " "
            decoded_word_length += 1
            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == ')' or
                    decoded_word_length > 49):
                stop_condition = True
            elif sampled_char.isdigit():
                decoded_location = sampled_char
            elif decoded_location != '':
                decoded_actions[int(decoded_location)] = sampled_char

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, 450), dtype=np.float32)
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = h
        print("decoded sentence ", decoded_sentence)
        return decoded_actions

    def decode_attention_sequence(self, model, input_seq, max_sequence_length):
        # Encode the input as state vectors.
        x, z_mean, z_log_var, z = model.encoder(input_seq)
        states_value = z
        context_vec, attention_W = model.attention([x, z])
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.dictionary_size), dtype=np.float32)
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, 448] = 1.

        stop_condition = False
        decoded_sentence = '( '
        decoded_word_length = 0
        decoded_actions = {}
        decoded_location = ''
        while not stop_condition:
            output_tokens, h = model.decoder(
                [target_seq, states_value, context_vec])

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.index_to_word_mapping[int(sampled_token_index)]
            decoded_sentence += sampled_char
            decoded_sentence += " "
            decoded_word_length += 1
            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == ')' or
                    decoded_word_length > 49):
                stop_condition = True
            elif sampled_char.isdigit():
                decoded_location = sampled_char
            elif decoded_location != '':
                decoded_actions[int(decoded_location)] = sampled_char

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, 450), dtype=np.float32)
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = h
        print("decoded sentence ", decoded_sentence)
        return decoded_actions

'''
########################################################################################################################
We don't use Version 1 now

'''

#
# class Halite(object):
#     """
#
#     """
#
#     def __init__(self):
#
#         self.replay = None
#         self.winner = -1
#         # halite per cell
#         self.halite = []
#         self.ship_actions = []
#         self.shipyard_actions = []
#         self.ship_position = []
#         self.shipyard_position = []
#         self.cargo = []
#         self.turns_left = []
#         self.spawn = []
#
#     # Converts position from 1D to 2D representation in (left most col, left most row)
#     def get_col_row(self, size: int, pos: int):
#         return pos % size, pos // size
#
#     # convert top left coordinate to (left row, left col)
#     def get_2D_col_row(self, size: int, pos: int):
#         top_left_row = pos // size
#         top_left_col = pos % size
#         return top_left_row, top_left_col
#
#     def get_to_pos(self, size: int, pos: int, direction: str):
#         col, row = self.get_col_row(size, pos)
#         if direction == "NORTH":
#             return pos - size if pos >= size else size ** 2 - size + col
#         elif direction == "SOUTH":
#             return col if pos + size >= size ** 2 else pos + size
#         elif direction == "EAST":
#             return pos + 1 if col < size - 1 else row * size
#         elif direction == "WEST":
#             return pos - 1 if col > 0 else (row + 1) * size - 1
#
#     def load_replay(self, path: str):
#         """
#                 load replay json file from halite website
#                 :return:
#                 """
#         # check there is a file in the given path
#         assert (os.path.isfile(path))
#         with open(path) as f:
#             self.replay = json.loads(f.read())
#
#     def load_data(self):
#         """
#         After replay is loaded, fetch raw data for training
#         :return:
#         """
#         assert (self.replay is not None)
#         # parameters that is used to retrieve specific piece of data
#         game_config = self.replay["configuration"]
#         number_of_players = len(self.replay["rewards"])
#         map_size = game_config["size"]
#         self.winner = self.find_winner()
#         self.halite = self.load_halite(map_size)
#         self.ship_actions, self.shipyard_actions, self.spawn = self.load_moves(map_size, number_of_players)
#         self.ship_position, self.cargo, self.shipyard_position = self.load_ship_shipyard_position(map_size)
#         total_step = self.cargo.shape[0]
#         self.turns_left = np.array([total_step - i for i in range(total_step)])
#
#     def find_winner(self):
#         assert (self.replay is not None)
#         assert (self.replay["steps"][-1] is not None)
#         player_reward = []
#         last_step = self.replay["steps"][- 1]
#         first_step = self.replay["steps"][1]
#         # has to be 4 player assume
#         assert (len(last_step) == len(first_step))
#         for player in last_step:
#             player_reward.append(player["reward"])
#         return player_reward.index(max(player_reward))
#
#     def load_halite(self, map_size: int):
#         """
#         Load the amount of halite on the map at each turn
#         :param map_size: the size of the map, assume the map is a square
#         :return: a 3D numpy array with dimension of steps - 1 x map_size x map_size
#         """
#         halite = []
#         assert (self.replay is not None)
#         for step, content in enumerate(self.replay["steps"]):
#             # Get board observation
#             if step == 0:
#                 continue
#             observation = self.replay["steps"][step - 1][0]["observation"]
#
#             # load the amount of halite on the map
#             raw_energy_grid = observation["halite"]
#             one_step = []
#             assert (len(raw_energy_grid) == map_size ** 2)
#             for i in range(map_size):
#                 one_row = []
#                 for j in range(map_size):
#                     one_row.append(raw_energy_grid[i * map_size + j])
#                 one_step.append(one_row)
#             halite.append(one_step)
#         halite = np.array(halite)
#         total_step = len(self.replay["steps"])
#         assert (halite.shape[0] == total_step - 1)
#         assert (halite.shape[1] == map_size)
#         assert (halite.shape[2] == map_size)
#         halite = halite / 100  # round it
#         # print(halite)
#         return halite
#
#     def load_moves(self, map_size: int, num_of_players: int):
#         """
#         Loads the ship actions and shipyard actions for player 0 at each turn
#         :return: a numpy 3D array
#         """
#         valid_actions = ["STAY", "EAST", "WEST", "SOUTH", "NORTH", "CONVERT"]
#         ships_action = []
#         shipyards_action = []
#         spawn = []
#         # Iterate through each step of the game to get step based information
#         for step, content in enumerate(self.replay["steps"]):
#             if step == 0:
#                 continue
#             # Get board observation
#             # observation = content[0]["observation"]
#             observation = self.replay["steps"][step - 1][0]["observation"]
#             # Declare processing information
#             # ship action and shipyard action per step
#             step_ships_action = np.zeros((map_size, map_size), np.int32)
#             step_shipyard_action = np.zeros((map_size, map_size), np.int32)
#             # ZERO if no ship yard spawned new ships
#             spawn.append(0)
#
#             # Load ship moves for all active players
#             for pid in range(len(content)):
#                 if "player" not in content[pid]["observation"]:
#                     continue
#                 player_id = content[pid]["observation"]["player"]
#
#                 # load player 0's information
#                 assert (self.winner != -1)
#                 if player_id == self.winner:
#                     player_observation = observation["players"][player_id]
#                     # Get halite, shipyard, ship information of the player
#                     player_shipyard = player_observation[1]
#                     player_ship = player_observation[2]
#                     # load action
#                     for ship_shipyard_id, move in content[pid]["action"].items():
#                         if move == "SPAWN":
#                             # check it is a shipyard
#                             assert (ship_shipyard_id in player_shipyard)
#                             # Get shipyard location
#                             shipyard_pos_1d = player_shipyard[ship_shipyard_id]
#                             shipyard_pos_2d = self.get_2D_col_row(map_size, shipyard_pos_1d)
#                             step_shipyard_action[shipyard_pos_2d[0]][shipyard_pos_2d[1]] = 1
#                             # update spawn to ONE since new ship spawned
#                             spawn[-1] = 1
#                         else:
#                             # check it is a valid move for ships
#                             assert (move in valid_actions)
#                             # get information of the ship
#                             ship_info = player_ship[ship_shipyard_id]
#                             assert (len(ship_info) == 2)  # [pos,cargo]
#                             ship_pos_1d = ship_info[0]
#                             ship_pos_2d = self.get_2D_col_row(map_size, ship_pos_1d)
#                             step_ships_action[ship_pos_2d[0]][ship_pos_2d[1]] = valid_actions.index(move)
#                     # print(step_ships_action)
#                     # print(step_shipyard_action)
#                     ships_action.append(step_ships_action)
#                     shipyards_action.append(step_shipyard_action)
#             # Load training features
#
#         ships_action = np.array(ships_action)
#         shipyards_action = np.array(shipyards_action)
#         spawn = np.array(spawn)
#         return ships_action, shipyards_action, spawn
#
#     def load_ship_shipyard_position(self, map_size):
#         """
#         Loads all active ships and shipyards 2D positions for all players at each turn,
#         the player of training will be loaded with 2, the rest will be 1
#         :return:
#         """
#         ships_position = []
#         ships_cargo = []
#         shipyard_position = []
#
#         # Iterate through each step of the game to get step based information
#         for step, content in enumerate(self.replay["steps"]):
#             if step == 0:
#                 continue
#             # Get board observation
#             # observation = content[0]["observation"]
#             observation = self.replay["steps"][step - 1][0]["observation"]
#
#             step_ships_position = np.zeros((map_size, map_size), np.int32)
#             step_shipyard_position = np.zeros((map_size, map_size), np.int32)
#             step_ship_cargo = np.zeros((map_size, map_size), np.float32)
#
#             # Load ship moves for all active players
#             for pid in range(len(content)):
#                 if "player" not in content[pid]["observation"]:
#                     continue
#                 player_id = content[pid]["observation"]["player"]
#
#                 # load player 0's information
#                 assert (self.winner != -1)
#                 player_observation = observation["players"][player_id]
#                 # Get halite, shipyard, ship information of the player
#                 player_shipyard = player_observation[1]
#                 player_ship = player_observation[2]
#                 # load ship position and halite carry amount
#                 for ship_id, ship_info in player_ship.items():
#                     assert (len(ship_info) == 2)  # ship_info : [pos,cargo]
#                     ship_pos_1d = ship_info[0]
#                     ship_pos_2d = self.get_2D_col_row(map_size, ship_pos_1d)
#                     step_ships_position[ship_pos_2d[0]][ship_pos_2d[1]] = 2 if player_id == self.winner else 1
#                     step_ship_cargo[ship_pos_2d[0]][ship_pos_2d[1]] = ship_info[1]
#
#                 # load shipyard position
#                 for shipyard_id, shipyard_info in player_shipyard.items():
#                     assert (isinstance(shipyard_info, int))  # shipyard_info : pos
#                     shipyard_pos_1d = shipyard_info
#                     shipyard_pos_2d = self.get_2D_col_row(map_size, shipyard_pos_1d)
#                     step_shipyard_position[shipyard_pos_2d[0]][
#                         shipyard_pos_2d[1]] = 2 if player_id == self.winner else 1
#             shipyard_position.append(step_shipyard_position)
#             ships_position.append(step_ships_position)
#             ships_cargo.append(step_ship_cargo)
#         ships_position = np.array(ships_position)
#         ships_cargo = np.array(ships_cargo) / 100
#         shipyard_position = np.array(shipyard_position)
#         return ships_position, ships_cargo, shipyard_position
#
#     def get_training_data(self, dim=32):
#         """
#         After loading the raw data, do further processing to get the data prepared for training
#         :param dim: the training data has the shape of [ None, dim, dim]
#         :return:
#         """
#         assert (self.replay is not None and self.halite is not None)
#         # print(self.halite.shape)
#         # print(self.ship_actions.shape)
#         assert (self.halite.shape[0] == self.ship_actions.shape[0])
#         step = self.halite.shape[0]
#         shape = self.halite.shape[1]
#         train_x = np.zeros((step, dim, dim, 4))
#         train_y_ship = np.zeros((step, dim, dim))
#         train_y_shipyard = np.zeros((step, dim, dim))
#         train_feature_mix = np.stack([self.halite, self.ship_position, self.cargo, self.shipyard_position], axis=-1)
#         # add padding
#         if shape != dim:
#             pad_offset = (dim - self.halite.shape[1]) // 2
#             train_x[:, pad_offset:pad_offset + shape, pad_offset:pad_offset + shape, :] = train_feature_mix
#             train_y_ship[:, pad_offset:pad_offset + shape, pad_offset:pad_offset + shape] = self.ship_actions
#             train_y_shipyard[:, pad_offset:pad_offset + shape, pad_offset:pad_offset + shape] = self.shipyard_actions
#             return train_x, train_y_ship, train_y_shipyard
#         return train_feature_mix, self.ship_actions.copy(), self.shipyard_actions.copy()
#
#     def get_my_ships(self, dim=32):
#         """
#                 After loading the raw data, get current player ship positions with padding
#                 :param dim: the training data has the shape of [ None, dim, dim]
#                 :return:
#                 """
#         assert (self.replay is not None and self.ship_position is not None)
#         step = self.ship_position.shape[0]
#         shape = self.ship_position.shape[1]
#         ship_info = np.zeros((step, dim, dim))
#         # add padding
#         if shape != dim:
#             pad_offset = (dim - self.halite.shape[1]) // 2
#             ship_info[:, pad_offset:pad_offset + shape, pad_offset:pad_offset + shape] = self.ship_position
#             return ship_info
#         return self.ship_position.copy()

# game = Halite()
# game.load_replay("1208740.json")
# #game.load_halite(21)
# game.load_data()
# game.get_training_data()
