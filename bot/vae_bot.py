from kaggle_environments.envs.halite.helpers import *
import numpy as np
import tensorflow as tf
from train import utils

class VaeBot(utils.Gameplay):
    """
    This bot inherits the base class for agent bot,
    it comes with a set of preprocessed data as well as helper function.
    For all the new data you added, you can reset them in the reset function,
    and scale them in the normalize function
    """
    def __init__(self, obs, config):
        super().__init__(obs, config)
        #self.vae, self.encoder, self.decoder = self.load_model()

    def reset_board(self, obs, config):
        super().reset_board(obs, config)

    def normalize(self):
        super().normalize()

    def load_model(self):
        vae = tf.keras.models.load_model('vae.h5')
        encoder = tf.keras.models.load_model('vae_encoder.h5')
        decoder = tf.keras.models.load_model('vae_decoder.h5')
        return vae, encoder, decoder

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
        if obs.step == 1:
            actions = {}
            this_turn = self
            current_player = this_turn.board.current_player
            for ship in current_player.ships:
                actions[ship.id] = "CONVERT"
            return actions
        elif obs.step == 2:
            actions = {}
            this_turn = self
            current_player = this_turn.board.current_player
            for shipyard in current_player.shipyards:
                actions[shipyard.id] = 'SPAWN'
            return actions
        # elif obs.step == 3:
        #     actions = {}
        #     this_turn = self
        #     current_player = this_turn.board.current_player
        #     for ship in current_player.ships:
        #         actions[ship.id] = "EAST"
        #     return actions

        this_turn = self
        current_player = this_turn.board.current_player
        size = 21
        pad_offset = 6

        input_image = np.zeros(
            (1, 32, 32, 5),
            dtype='float32')
        # Load halite

        for i in range(size):
            for j in range(size):
                input_image[0][i + pad_offset][j + pad_offset][0] = obs.halite[i * size + j] / 10
        # Load current player and cargo
        for ship in current_player.ships:
            position = self.convert_kaggle2D_to_upperleft2D(this_turn.board_size, list(ship.position))
            input_image[0][position[0] + pad_offset][position[1] + pad_offset][1] = 10.0
            cargo = self.my_cargo[position[0]][position[1]] / 10
            print("cargo amount at", cargo)
            input_image[0][position[0] + pad_offset][position[1] + pad_offset][2] = cargo
        # 4) ship yard

        for shipyard in current_player.shipyards:
            position = self.convert_kaggle2D_to_upperleft2D(this_turn.board_size, list(shipyard.position))
            input_image[0][position[0]+pad_offset][position[1]+pad_offset][3] = 10.0

        #Other player ship
        other_players = self.board.opponents
        # 5) other players' ship
        for player in other_players:
            for ship in player.ships:
                position = self.convert_kaggle2D_to_upperleft2D(size, list(ship.position))
                input_image[0][position[0] + pad_offset][position[1] + pad_offset][4] = 1 * 10
        valid_move = ["STAY", "EAST", "WEST", "SOUTH", "NORTH", "CONVERT"]
        # Define sampling models

        vae = tf.saved_model.load('vae_new')
        num_dict = {}
        for i in range(size ** 2):
            num_dict[i] = str(i)
        vocab_idx = size ** 2
        move_option = ["EAST", "WEST", "SOUTH", "NORTH", "CONVERT", "SPAWN", "NO", "(", ")"]
        for option in move_option:
            num_dict[vocab_idx] = option
            vocab_idx += 1
        def decode_sequence(input_seq):
            # Encode the input as state vectors.

            z_mean, z_log_var, z = vae.encoder(input_seq)
            states_value = z
            # Generate empty target sequence of length 1.
            target_seq = np.zeros((1, 1, 450),dtype=np.float32)
            # Populate the first character of target sequence with the start character.
            target_seq[0, 0, 448] = 1.
            #target_seq = np.float32(target_seq)
            # Sampling loop for a batch of sequences
            # (to simplify, here we assume a batch of size 1).
            stop_condition = False
            decoded_sentence = '( '
            decoded_word_length = 0
            decoded_actions = {}
            decoded_location = ''
            while not stop_condition:
                output_tokens, h = vae.decoder(
                    [target_seq, states_value])

                # Sample a token
                sampled_token_index = np.argmax(output_tokens[0, -1, :])
                sampled_char = num_dict[int(sampled_token_index)]
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
                target_seq = np.zeros((1, 1, 450),dtype=np.float32)
                target_seq[0, 0, sampled_token_index] = 1.

                # Update states
                states_value = h
            print("decoded sentence ", decoded_sentence)
            return decoded_actions

        def decode_sequence_fixed_state(input_seq):
            # Encode the input as state vectors.
            z_mean, z_log_var, z = vae.encoder(input_seq)
            states_value = z
            # Generate empty target sequence of length 1.
            target_seq = np.zeros((1, 1, 450),dtype=np.float32)
            # Populate the first character of target sequence with the start character.
            target_seq[0, 0, 448] = 1.
            #target_seq = np.float32(target_seq)
            # Sampling loop for a batch of sequences
            # (to simplify, here we assume a batch of size 1).
            stop_condition = False
            decoded_sentence = ''
            while not stop_condition:
                output_tokens, h = vae.decoder(
                    [target_seq, states_value])

                # Sample a token
                sampled_token_index = np.argmax(output_tokens[0, -1, :])
                sampled_char = num_dict[int(sampled_token_index)]
                decoded_sentence += sampled_char
                decoded_sentence += " "
                # Exit condition: either hit max length
                # or find stop character.
                if (sampled_char == ')' or
                        len(decoded_sentence) > 49):
                    stop_condition = True

                # Update the target sequence (of length 1).
                target_seq = np.zeros((1, 1, 450), dtype=np.float32)
                target_seq[0, 0, sampled_token_index] = 1.

            return decoded_sentence
        result = decode_sequence(input_image)
        print("decoded actions ", result)

        # Assign actions to shipyard or ships with exact location matching
        actions = {}
        for shipyard in current_player.shipyards:
            position = self.convert_kaggle2D_to_kaggle1D(size, shipyard.position)
            print("shipyard position is", position)
            if position in result:
                if result[position] == 'SPAWN':
                    actions[shipyard.id] = result[position]
                    del result[position]
        unassigned_ships = {}
        for ship in current_player.ships:
            position = self.convert_kaggle2D_to_kaggle1D(size, ship.position)
            print("ship position is", position)
            if position in result and result[position] != 'SPAWN':
                if result[position] != 'NO':
                    actions[ship.id] = result[position]
                del result[position]
            else:
                unassigned_ships[ship.id] = position

        print("unassigned actions", result)
        print("unassigned ships", unassigned_ships)

        # Match unassigned ships with actions in the ships proximity
        for id, pos in unassigned_ships.items():
            valid_action = self.find_actions_in_ship_proximity(size, result, pos)
            if valid_action:
                if valid_action != 'NO':
                    actions[id] = valid_action
                    print("assigned valid action", valid_action)

        return actions
