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
        self.vae_encoder_input_image = self.prepare_encoder_input()

    def reset_board(self, obs, config):
        super().reset_board(obs, config)

    def normalize(self):
        """
        :return:
        """
        pass

    def load_model(self):
        vae = tf.keras.models.load_model('vae.h5')
        encoder = tf.keras.models.load_model('vae_encoder.h5')
        decoder = tf.keras.models.load_model('vae_decoder.h5')
        return vae, encoder, decoder

    def prepare_encoder_input(self):

        """
        Currently, we have 5 features
        Four features as training input:
            1) halite available
            2) my ship
            3) cargo on my ship
            4) my shipyard
            5) other players' ships
        """

        this_turn = self
        current_player = this_turn.board.current_player

        size = 21
        pad_offset = 6
        num_of_encoder_features = 5
        obs = self.obs
        input_image = np.zeros(
            (1, 32, 32, num_of_encoder_features),
            dtype='float32')


        # Load halite
        for i in range(size):
            for j in range(size):
                input_image[0][i + pad_offset][j + pad_offset][0] = obs["halite"][i * size + j] * 10 / 100
        # Load current player and cargo
        for ship in current_player.ships:
            position = self.convert_kaggle2D_to_upperleft2D(this_turn.board_size, list(ship.position))
            input_image[0][position[0] + pad_offset][position[1] + pad_offset][1] = 10.0
            cargo = self.my_cargo[position[0]][position[1]] * 10
            input_image[0][position[0] + pad_offset][position[1] + pad_offset][2] = cargo * 10 / 100
        # 4) ship yard

        for shipyard in current_player.shipyards:
            position = self.convert_kaggle2D_to_upperleft2D(this_turn.board_size, list(shipyard.position))
            input_image[0][position[0] + pad_offset][position[1] + pad_offset][3] = 10.0
        actions = {}

        # Other player ship
        other_players = self.board.opponents
        # 5) other players' ship
        for player in other_players:
            for ship in player.ships:
                position = self.convert_kaggle2D_to_upperleft2D(size, list(ship.position))
                input_image[0][position[0] + pad_offset][position[1] + pad_offset][4] = 1 * 10
        return input_image
        # Define sampling models
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
        actions = {}
        # input_image = np.zeros(
        #     (1, 32, 32, 5),
        #     dtype='float32')
        # # Load halite
        #
        # for i in range(size):
        #     for j in range(size):
        #         input_image[0][i + pad_offset][j + pad_offset][0] = obs.halite[i * size + j] * 10/100
        # # Load current player and cargo
        # for ship in current_player.ships:
        #     position = self.convert_kaggle2D_to_upperleft2D(this_turn.board_size, list(ship.position))
        #     input_image[0][position[0] + pad_offset][position[1] + pad_offset][1] = 10.0
        #     cargo = self.my_cargo[position[0]][position[1]] * 10
        #     input_image[0][position[0] + pad_offset][position[1] + pad_offset][2] = cargo * 10/100
        # # 4) ship yard
        #
        # for shipyard in current_player.shipyards:
        #     position = self.convert_kaggle2D_to_upperleft2D(this_turn.board_size, list(shipyard.position))
        #     input_image[0][position[0]+pad_offset][position[1]+pad_offset][3] = 10.0
        #
        #
        # #Other player ship
        # other_players = self.board.opponents
        # # 5) other players' ship
        # for player in other_players:
        #     for ship in player.ships:
        #         position = self.convert_kaggle2D_to_upperleft2D(size, list(ship.position))
        #         input_image[0][position[0] + pad_offset][position[1] + pad_offset][4] = 1 * 10
        # Define sampling models
        size = self.board_size
        vae = tf.saved_model.load('vae_new')
        input_image = self.vae_encoder_input_image
        inference_decoder = utils.Inference(board_size=21)
        result = inference_decoder.decode_sequence(vae, input_image, 50)
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
