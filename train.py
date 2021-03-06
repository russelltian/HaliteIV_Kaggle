import pickle
import gzip
import time
import random
from random import shuffle
# from multiprocessing import Process, Queue
from threading import Thread
from queue import Queue

import numpy as np
import tensorflow._api.v2.compat.v1 as tf

from train import utils
import os
#from core.data_utils import Game

from architecture import build_model
PATH = 'train/top_replay/'
replay_files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(PATH):
    for file in f:
        if '.json' in file:
            replay_files.append(os.path.join(r, file))
for f in replay_files:
    print(f)
queue = [Queue(32)]
queue_m_sizes = [32]

batch_size = len(replay_files)


#exit(0)

build_model()
frames_node = tf.get_collection('frames')[0]
# can_afford_node = tf.get_collection('can_afford')[0]
turns_left_node = tf.get_collection('turns_left')[0]
my_ships_node = tf.get_collection('my_ships')[0]
moves_node = tf.get_collection('moves')[0]
spawn_node = tf.get_collection('spawn')[0]
loss_node = tf.get_collection('loss')[0]
optimizer_node = tf.get_collection('optimizer')[0]

saver = tf.train.Saver(max_to_keep=1)
# path = random.choice(replay_files)
# # Load all training data
# game = utils.Halite()
# #path = '1068739.json'
# game.load_replay(path)
# game.load_data()
# print("winner of this game is player: ", game.winner)
# X_frame, Y_ship, Y_shipyard = game.get_training_data()
# X_ship = game.get_my_ships()
# turns_left = game.turns_left
#assert (turns_left is not None)
with tf.Session() as sess:
    tf.initializers.global_variables().run()

    for step in range(5):
        # Load all training data
        game = utils.Halite()
        # path = random.choice(replay_files)
        game.load_replay(path)
        game.load_data()
        X_frame, Y_ship, Y_shipyard = game.get_training_data()
        X_ship = game.get_my_ships()
        turns_left = game.turns_left
        spawn = game.spawn
        assert (turns_left is not None)


        # first batch of parameters X: (halite,ship_pos) Y: ship_moves
        f_batch, m_batch, spawn_batch = [], [], []
        print(step)
        # second batch of parameters: ship_pos
        s_batch = []
        # third batch of parameters: turns_left
        t_batch = []
        #total_size = X_frame.shape[0]
        for i in range(batch_size):

            #rand_num = random.randint(0, total_size-1)
            path = replay_files[i]
            rand_num = random.randint(1, 398)
            # Load all training data
            game = utils.Halite()
            game.load_replay(path)
            game.load_data()
            print("winner of this game is player: ", game.winner)
            X_frame, Y_ship, Y_shipyard = game.get_training_data()
            X_ship = game.get_my_ships()
            turns_left = game.turns_left
            assert (turns_left is not None)
            frame, move, generate = X_frame[rand_num], Y_ship[rand_num], Y_shipyard[rand_num]

            my_ships = X_ship[rand_num]
            turn_left = turns_left[rand_num].reshape(1)
            generate = spawn[rand_num].reshape(1)
            f_batch.append(frame)
            m_batch.append(move)
            spawn_batch.append(generate)
            # c_batch.append(can_afford)
            t_batch.append(turn_left)
            s_batch.append(my_ships)

        f_batch = np.stack(f_batch)
        m_batch = np.stack(m_batch)
        spawn_batch = np.stack(spawn_batch)
        # c_batch = np.stack(c_batch)
        t_batch = np.stack(t_batch)
        s_batch = np.stack(s_batch)

        # spawn_batch = np.expand_dims(spawn_batch, -1)
        # t_batch = np.expand_dims(t_batch, -1)
        m_batch = np.expand_dims(m_batch, -1)
        # f_batch = np.expand_dims(f_batch, -1)
        s_batch = np.expand_dims(s_batch, -1)

        # print([x.shape for x in [f_batch, m_batch]])

        # feed_dict = {frames_node: f_batch,
        #              can_afford_node: c_batch,
        #              turns_left_node: t_batch,
        #              my_ships_node: s_batch,
        #              moves_node: m_batch,
        #              generate_node: g_batch,
        #              }

        feed_dict = {frames_node: f_batch,
                     moves_node: m_batch,
                     spawn_node: spawn_batch,
                     turns_left_node: t_batch,
                     my_ships_node: s_batch,
                     }

        for i in range(100):
            loss, _ = sess.run([loss_node, optimizer_node], feed_dict=feed_dict)
            print(loss)
    #saver.save(sess, os.path.join("./", 'model_{}.ckpt'.format(step)))
    saver.save(sess, os.path.join("./", 'model.ckpt'))
    writer = tf.summary.FileWriter('./', sess.graph)
        # val = queue.get()
        # print(val)

# Probably want to mean by ship number and weight other factors, like time
# step in game to balance the training.