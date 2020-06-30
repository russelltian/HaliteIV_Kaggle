import copy
import json
import os
import numpy as np

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
        valid_moves = ["NORTH", "EAST", "SOUTH", "WEST"]
        for step, content in enumerate(replay["steps"]):
            step_move = np.zeros((map_size, map_size, num_players))
            observation = content[0]["observation"]
            # for pid in range(num_players):
            #     # TODO: check if all 4 players in observation
            #     print(players[pid])
            for pid in range(len(content)):
                if "player" not in content[pid]["observation"]:
                    continue
                player_id = content[pid]["observation"]["player"]

                #for object_id, move in content[pid]["action"].items():
                    #print(object_id, move)

            # load halite energy
            raw_energy_grid = observation["halite"]
            oneframe = []
            assert(len(raw_energy_grid) == map_size**2)
            for i in range(map_size):
                onerow = []
                for j in range(map_size):
                    onerow.append(raw_energy_grid[j])
                oneframe.append(onerow)
            #print(oneframe)
            oneframe = np.array(oneframe)
            print(oneframe)
    load_moves()


    #print(replay["steps"][-1])

load_replay("1208740.json")