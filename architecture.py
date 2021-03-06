import tensorflow._api.v2.compat.v1 as tf

tf.compat.v1.disable_eager_execution()
def build_model():

    size = 8  # Single size for easier debugging (for now)
    max_s = [1, 2, 2, 1]  # size of the sliding window for max pooling
    learning_rate = 0.0001

    # frames = tf.placeholder(tf.float32, [None, 256, 256, 5]) # None is the number of samples, rename the variable name later
    frames = tf.placeholder(tf.float32, [None, 32, 32, 4], name="frames") # features: halite_available, others_ship, cargo, self_shipyard
    # can_afford = tf.placeholder(tf.float32, [None, 3])
    turns_left = tf.placeholder(tf.float32, [None, 1], name="turnsleft")
    my_ships = tf.placeholder(tf.float32, [None, 32, 32, 1], name="myships")

    my_ships = tf.cast(my_ships, tf.float32)

    moves = tf.placeholder(tf.uint8, [None, 32, 32, 1], name="moves")
    spawn = tf.placeholder(tf.float32, [None, 1], name="spawn")

    tf.add_to_collection('frames', frames)
    # tf.add_to_collection('can_afford', can_afford)
    tf.add_to_collection('turns_left', turns_left)
    tf.add_to_collection('my_ships', my_ships)
    tf.add_to_collection('moves', moves)
    tf.add_to_collection('spawn', spawn)

    moves = tf.one_hot(moves, 6)

    # ca = tf.layers.dense(can_afford, size)
    tl = tf.layers.dense(turns_left, size)

    # ca = tf.expand_dims(ca, 1)
    # ca = tf.expand_dims(ca, 1)
    tl = tf.expand_dims(tl, 1)
    tl = tf.expand_dims(tl, 1)

    d_l1_a = tf.layers.conv2d(frames, size, 3, activation=tf.nn.relu, padding='same')  # input is frames, filters is size, kernal size is 3(x3)
    d_l1_p = tf.nn.max_pool(d_l1_a, max_s, max_s, padding='VALID')  # 16

    d_l2_a = tf.layers.conv2d(d_l1_p, size, 3, activation=tf.nn.relu, padding='same')
    d_l2_p = tf.nn.max_pool(d_l2_a, max_s, max_s, padding='VALID')  # 8

    d_l3_a = tf.layers.conv2d(d_l2_p, size, 3, activation=tf.nn.relu, padding='same')
    d_l3_p = tf.nn.max_pool(d_l3_a, max_s, max_s, padding='VALID')  # 4

    d_l4_a = tf.layers.conv2d(d_l3_p, size, 3, activation=tf.nn.relu, padding='same')
    d_l4_p = tf.nn.max_pool(d_l4_a, max_s, max_s, padding='VALID')  # 2

    d_l5_a = tf.layers.conv2d(d_l4_p, size, 3, activation=tf.nn.relu, padding='same')
    d_l5_p = tf.nn.max_pool(d_l5_a, max_s, max_s, padding='VALID')  # 1

    final_state = tf.concat([d_l5_p, tl], -1)
    latent = tf.layers.dense(final_state, size, activation=tf.nn.relu)
    # latent = tf.layers.dense(d_l5_p, size, activation=tf.nn.relu)

    u_l5_a = tf.layers.conv2d_transpose(latent, size, 3, 2, activation=tf.nn.relu, padding='same')  # 2
    u_l5_c = tf.concat([u_l5_a, d_l5_a], -1)
    u_l5_s = tf.layers.conv2d(u_l5_c, size, 3, activation=tf.nn.relu, padding='same')

    u_l4_a = tf.layers.conv2d_transpose(u_l5_s, size, 3, 2, activation=tf.nn.relu, padding='same')  # 4
    u_l4_c = tf.concat([u_l4_a, d_l4_a], -1)
    u_l4_s = tf.layers.conv2d(u_l4_c, size, 3, activation=tf.nn.relu, padding='same')

    u_l3_a = tf.layers.conv2d_transpose(u_l4_s, size, 3, 2, activation=tf.nn.relu, padding='same')  # 8
    u_l3_c = tf.concat([u_l3_a, d_l3_a], -1)
    u_l3_s = tf.layers.conv2d(u_l3_c, size, 3, activation=tf.nn.relu, padding='same')

    u_l2_a = tf.layers.conv2d_transpose(u_l3_s, size, 3, 2, activation=tf.nn.relu, padding='same')  # 16
    u_l2_c = tf.concat([u_l2_a, d_l2_a], -1)
    u_l2_s = tf.layers.conv2d(u_l2_c, size, 3, activation=tf.nn.relu, padding='same')

    u_l1_a = tf.layers.conv2d_transpose(u_l2_s, size, 3, 2, activation=tf.nn.relu, padding='same')  # 32
    u_l1_c = tf.concat([u_l1_a, d_l1_a], -1)
    u_l1_s = tf.layers.conv2d(u_l1_c, size, 3, activation=tf.nn.relu, padding='same')

    spawn_logits = tf.layers.dense(latent, 1, activation=None)
    #
    spawn_logits = tf.squeeze(spawn_logits, [1, 2])

    moves_logits = tf.layers.conv2d(u_l1_s, 6, 3, activation=None, padding='same')

    tf.add_to_collection('m_logits', moves_logits)
    tf.add_to_collection('s_logits', spawn_logits)

    losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=moves,
                                                        logits=moves_logits,
                                                        dim=-1)

    losses = tf.expand_dims(losses, -1)

    masked_loss = losses * my_ships

    ships_per_frame = tf.reduce_sum(my_ships, axis=[1, 2])

    frame_loss = tf.reduce_sum(masked_loss, axis=[1, 2])

    average_frame_loss = frame_loss / (ships_per_frame + 0.00000001) # First frames have no ship

    spawn_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=spawn, logits=spawn_logits)

    spawn_losses = tf.reduce_mean(spawn_losses)

    loss = tf.reduce_mean(average_frame_loss) + 0.01 * spawn_losses

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    tf.add_to_collection('loss', loss)
    tf.add_to_collection('optimizer', optimizer)

    return