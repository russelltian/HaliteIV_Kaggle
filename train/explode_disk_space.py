import os
from pathlib import Path
from train import utils
import numpy as np



training_datasets = []
def save_training_data():
    from numpy import save
    PATH = 'top_replay/'
    replay_files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(PATH):
        for file in f:
            if '.json' in file:
                replay_files.append(os.path.join(r, file))
    for f in replay_files:
        print(f)
    game = None
    # tables.open_file

    for i, path in enumerate(replay_files):
        game = utils.HaliteV2(path)
        print("index", i)
        # if i == 3:
        #     break
        if game.game_play_list is not None and game.winner_id == 0:
            game.prepare_data_for_vae()
            training_input = np.zeros(
                (400, 32, 32, 4),
                dtype='float32')

            my_ship_positions = game.ship_position
            target_ship_actions = game.ship_actions
            halite_available = game.halite
            my_shipyard = game.shipyard_position
            my_cargo = game.cargo

            """
            Target ship actions:
            """
            training_label = np.zeros(
                (400, 32, 32, 6),
                dtype='float32')

            pad_offset = 6

            #  1) halite available
            for i, halite_map in enumerate(zip(halite_available)):
                # print("halite_map", halite_map)
                for row_indx, row in enumerate(halite_map[0]):
                    row = np.squeeze(row)
                    for col_indx, item in enumerate(row):
                        # print(item)
                        training_input[i, row_indx + pad_offset, col_indx + pad_offset, 0] = item * 10

            # 2) my ship position
            for i, my_ship_position in enumerate(my_ship_positions):
                for row_indx, row in enumerate(my_ship_position):
                    for col_indx, item in enumerate(row):
                        training_input[i, row_indx + pad_offset, col_indx + pad_offset, 1] = item * 10

            # 3) cargo on my ship
            for i, cargo_map in enumerate(my_cargo):
                for row_indx, row in enumerate(cargo_map):
                    for col_indx, item in enumerate(row):
                        training_input[i, row_indx + pad_offset, col_indx + pad_offset, 2] = item * 10

            # 4) my ship yard position
            for i, shipyard_map in enumerate(my_shipyard):
                for row_indx, row in enumerate(shipyard_map):
                    for col_indx, item in enumerate(row):
                        training_input[i, row_indx + pad_offset, col_indx + pad_offset, 3] = item * 10

            # target actions
            for i, target_ship_action in enumerate(target_ship_actions):
                for row_indx, row in enumerate(target_ship_action):
                    for col_indx, item in enumerate(row):
                        training_label[i, row_indx + pad_offset, col_indx + pad_offset, int(item)] = 1.

            print("training input shape", training_input.shape)

            # Do word embedding
            board_size = game.config["size"]
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
            # target actions
            decoder_input_data = np.zeros(
                (400, 50, len(vocab_dict)),
                dtype='float32')
            decoder_target_data = np.zeros(
                (400, 50, len(vocab_dict)),
                dtype='float32')

            sequence = game.move_sequence
            sequence.append(sequence[-1])
            # TODO: validate max sequence
            for step, each_sequence in enumerate(sequence):
                each_sequence_list = each_sequence.split()
                idx = 0
                last_word = ""
                for each_word in each_sequence_list:
                    assert (each_word in vocab_dict)
                    # TODO: Bug index > 50
                    if idx == 49:
                        break
                    assert (idx < 50)
                    if idx == 0:
                        decoder_input_data[step][idx][-2] = 1.
                        decoder_target_data[step][idx][vocab_dict[each_word]] = 1.
                    else:
                        decoder_input_data[step][idx][vocab_dict[last_word]] = 1.
                        decoder_target_data[step][idx][vocab_dict[each_word]] = 1.
                    idx += 1
                    last_word = each_word

                decoder_input_data[step][idx][vocab_dict[last_word]] = 1.
                decoder_target_data[step][idx][-1] = 1.
            print("target action shape", decoder_target_data.shape)
            # data_tensor = tf.convert_to_tensor(decoder_input_data)
            # data_tensor2 = tf.convert_to_tensor(decoder_target_data)

            # train_dataset = tf.data.Dataset.from_tensor_slices(([training_input, decoder_input_data], decoder_target_data))
            # save('encoder_input.npy', training_input)
            # save('decoder_input.npy', decoder_input_data)
            # save('decoder_target.npy', decoder_target_data)
            filename = str(path.split('.')[0])

            p = Path(filename)
            np.savez(filename, encoder_input=training_input, decoder_input=decoder_input_data,
                    decoder_output=decoder_target_data)#[training_input, decoder_input_data, decoder_target_data])
            #print("dataset shape", len(list(train_dataset.as_numpy_iterator())))
def load_file(path = 'training_data.npy'):
    PATH = 'top_replay/'
    replay_files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(PATH):
        for file in f:
            if '.npz' in file:
                replay_files.append(os.path.join(r, file))
    for path in replay_files:
        data = np.load(path)
        print(data['encoder_input'].shape)
        print(data['decoder_input'].shape)
        print(data['decoder_output'].shape)

save_training_data()
load_file()