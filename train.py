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

queue = [Queue(32)]
queue_m_sizes = [32]

batch_size = 10

# Load all training data
game = utils.Halite()
path = "./train/1208740.json"
game.load_replay(path)
game.load_data()
X_frame, Y = game.get_training_data()
X_ship = game.get_my_ships()
turns_left = game.turns_left
# print(type(X))
# print(Y.shape)
#exit(0)

build_model()

frames_node = tf.get_collection('frames')[0]
# can_afford_node = tf.get_collection('can_afford')[0]
turns_left_node = tf.get_collection('turns_left')[0]
my_ships_node = tf.get_collection('my_ships')[0]
moves_node = tf.get_collection('moves')[0]
# generate_node = tf.get_collection('generate')[0]
loss_node = tf.get_collection('loss')[0]
optimizer_node = tf.get_collection('optimizer')[0]

saver = tf.train.Saver(max_to_keep=1)

with tf.Session() as sess:
    tf.initializers.global_variables().run()

    for step in range(10):
        # first batch of parameters X: (halite,ship_pos) Y: ship_moves
        f_batch, m_batch = [], []
        print(step)
        # second batch of parameters: ship_pos
        s_batch = []
        # third batch of parameters: turns_left
        t_batch = []
        total_size = X_frame.shape[0]
        for i in range(batch_size):
            rand_num = random.randint(0, total_size-1)
            frame, move = X_frame[rand_num], Y[rand_num]
            my_ships = X_ship[rand_num]
            turn_left = np.append([], turns_left[rand_num])
            f_batch.append(frame)
            m_batch.append(move)
            # g_batch.append(generate)
            # c_batch.append(can_afford)
            t_batch.append(turn_left)
            s_batch.append(my_ships)
        f_batch = np.stack(f_batch)
        m_batch = np.stack(m_batch)
        # g_batch = np.stack(g_batch)
        # c_batch = np.stack(c_batch)
        t_batch = np.stack(t_batch)
        s_batch = np.stack(s_batch)

        # g_batch = np.expand_dims(g_batch, -1)
       # t_batch = np.expand_dims(t_batch, -1)
        m_batch = np.expand_dims(m_batch, -1)
        #f_batch = np.expand_dims(f_batch, -1)
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
                     turns_left_node: t_batch,
                     my_ships_node: s_batch,
                     }

        for i in range(200):
            loss, _ = sess.run([loss_node, optimizer_node], feed_dict=feed_dict)
            print(loss)
    #saver.save(sess, os.path.join("./", 'model_{}.ckpt'.format(step)))
    saver.save(sess, os.path.join("./", 'model.ckpt'))
        # val = queue.get()
        # print(val)

# Probably want to mean by ship number and weight other factors, like time
# step in game to balance the training.