import copy
import json
import os
import numpy as np
import sys
sys.path.append("../train")
from train import geometry

def load_replay(path: str):
    """
    load replay json file from halite website
    :return:
    """
    # check there is a file in the given path
    assert(os.path.isfile(path))

    with open(path) as f:
        replay = json.loads(f.read())

    game_config = replay["configuration"]
    number_of_players = len(replay["rewards"])

    def load_moves():
        map_size = game_config["size"]
        num_players = number_of_players
        total_steps = len(replay["steps"])
        valid_moves = ["NORTH", "EAST", "SOUTH", "WEST"]
        player0 = []
        # Iterate through each step of the game to get frame based information
        for step, content in enumerate(replay["steps"]):
            step_move = np.zeros((map_size, map_size, num_players))
            player0_ship = np.zeros((map_size,map_size))
            # Get board observation
            observation = content[0]["observation"]

            # Load ship moves for all active players
            for pid in range(len(content)):
                if "player" not in content[pid]["observation"]:
                    continue
                player_id = content[pid]["observation"]["player"]

                #for object_id, move in content[pid]["action"].items():
                    #print(object_id, move)

                # view of whole board at this step
                # load player 0's ship location
                if player_id == 0:
                    player_observation = observation["players"][player_id]
                    # Get halite, shipyard, ship information of the player
                    player_halite = player_observation[0]
                    player_shipyard = player_observation[1]
                    player_ship = player_observation[2]

                    # store ship information on the board
                    for ship_id, ship_info in player_ship.items():
                        assert(len(ship_info) == 2) # [pos,cargo]
                        ship_pos_1d = ship_info[0]
                        ship_halite = ship_info[1]
                        ship_pos_2d = geometry.get_2D_col_row(map_size,ship_pos_1d)
                        player0_ship[ship_pos_2d[0]][ship_pos_2d[1]] = 1
                    player0 = np.append(player0,player0_ship)
            print(player0.shape)
            # load halite energy
            raw_energy_grid = observation["halite"]
            oneframe = []
            assert(len(raw_energy_grid) == map_size**2)
            for i in range(map_size):
                onerow = []
                for j in range(map_size):
                    onerow.append(raw_energy_grid[j])
                oneframe.append(onerow)
            oneframe = np.array(oneframe)
            #print(oneframe)
        player0 = player0.reshape((map_size,map_size,total_steps))
        print(player0.shape)
        return player0
    load_moves()


    #print(replay["steps"][-1])

load_replay("1208740.json")