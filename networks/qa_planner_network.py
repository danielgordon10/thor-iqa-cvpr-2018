import numpy as np
import tensorflow as tf
from utils import tf_util
from networks.rl_network import A3CNetwork
from networks.free_space_network import FreeSpaceNetwork

import constants

slim = tf.contrib.slim


class QAPlannerNetwork(FreeSpaceNetwork, A3CNetwork):
    def create_net(self):
        self.episode_length_placeholder = tf.placeholder(tf.float32, [self.batch_size, None], name='episode_length_placeholder')
        self.question_count_placeholder = tf.placeholder(tf.float32, [self.batch_size, None], name='question_count_placeholder')
        self.conv_layers()
        self.fc_layers()
        self.gru_layer()
        self.map_mask_placeholder = tf.placeholder(tf.float32, [self.batch_size, None, constants.SPATIAL_MAP_HEIGHT, constants.SPATIAL_MAP_WIDTH, 6 + constants.NUM_CLASSES],
                                                   name='map_mask_placeholder')
        self.possible_move_placeholder = tf.placeholder(tf.float32, [self.batch_size, None, constants.STEPS_AHEAD ** 2 + 6], name='possible_move_placeholder')
        self.rl_layers()
        self.rl_loss()
        self.answer_loss()
        self.move_loss()

    def conv_layers(self):
        inputs = (tf.to_float(tf_util.remove_axis(self.image_placeholder, 1)) - 128) / 128

        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.elu):
            net = slim.conv2d(inputs, 32, [7, 7], stride=2, scope='conv1')
            net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool1', padding='SAME')
            net = slim.conv2d(net, 64, [5, 5], stride=1, scope='conv2')
            net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool2', padding='SAME')
            net = slim.conv2d(net, 128, [3, 3], stride=1, scope='conv3')
            net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool3', padding='SAME')
            net = slim.conv2d(net, 256, [3, 3], stride=1, scope='conv4')
            net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool4', padding='SAME')
        self.conv_output = net

    def fc_layers(self):
        net = self.conv_output
        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.elu):
            net = tf_util.remove_axis(net, [2, 3])
            net = slim.fully_connected(net, 1024, scope='fc1')
            self.meta_action_placeholder = tf.placeholder(tf.float32, [None, None, 7], name='meta_action_placeholder')
            net = tf.concat((net, tf_util.remove_axis(self.meta_action_placeholder, 1)), axis=1)
        self.fc_output = net

    def rl_layers(self):
        # Get score for each possible location.
        with tf.variable_scope('rl_layers'):
            # Add context
            with tf.variable_scope('question_embed'):
                self.question_object_placeholder = tf.placeholder(tf.int32, [self.batch_size, None], name='question_object_placeholder')
                self.question_object_one_hot = tf.one_hot(self.question_object_placeholder[:, 0], constants.NUM_CLASSES)

                self.question_type_placeholder = tf.placeholder(tf.int32, [self.batch_size, None], name='question_type_placeholder')
                self.question_type_one_hot = tf.one_hot(self.question_type_placeholder[:, 0], 4)

                self.question_container_placeholder = tf.placeholder(tf.float32, [self.batch_size, None, constants.NUM_CLASSES], name='question_container_placeholder')
                self.question_container_one_hot = self.question_container_placeholder[:, 0, :]

                self.question_direction_placeholder = tf.placeholder(tf.float32, [self.batch_size, None, 4], name='question_direction_placeholder')
                self.question_direction_one_hot = self.question_direction_placeholder[:, 0, :]

                with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.elu):
                    question_object_embedding = slim.fully_connected(self.question_object_one_hot, 64, scope='question')
                    question_type_embedding = slim.fully_connected(self.question_type_one_hot, 8, scope='question_type')
                    question_container_embedding = slim.fully_connected(self.question_container_one_hot, 64, scope='question_container')
                    question_direction_embedding = slim.fully_connected(self.question_direction_one_hot, 8, scope='question_direction')
                    question_embedding = tf.concat((question_object_embedding, question_type_embedding, question_container_embedding, question_direction_embedding), axis=1)

            with tf.variable_scope('spatial_map'):
                self.crop_size = 81
                rf_pad = int(np.floor(self.crop_size / 2))

                with tf.variable_scope('crop'):
                    def spatial_crop(memory_chunk, pose):
                        with tf.variable_scope('action_planner_memory_crop'):
                            start_x = tf.clip_by_value(pose[0] - rf_pad, 0, constants.SPATIAL_MAP_WIDTH)
                            end_x = tf.clip_by_value(pose[0] + rf_pad + 1, 0, constants.SPATIAL_MAP_WIDTH)
                            start_y = tf.clip_by_value(pose[1] - rf_pad, 0, constants.SPATIAL_MAP_HEIGHT)
                            end_y = tf.clip_by_value(pose[1] + rf_pad + 1, 0, constants.SPATIAL_MAP_HEIGHT)

                            memory_crop = memory_chunk[start_y:end_y, start_x:end_x, :]

                            memory_crop = tf.pad(memory_crop,
                                                 ((tf.maximum(0, start_y - (pose[1] - rf_pad)),
                                                   tf.maximum(0, (pose[1] + rf_pad + 1) - end_y)),
                                                  (tf.maximum(0, start_x - (pose[0] - rf_pad)),
                                                   tf.maximum(0, (pose[0] + rf_pad + 1) - end_x)),
                                                  (0, 0)))
                            memory_crop = tf.reshape(memory_crop, (self.crop_size, self.crop_size, memory_chunk.get_shape().as_list()[-1]))
                            memory_crop_rot = tf.cond(tf.equal(pose[2], 0),
                                                      lambda: memory_crop,
                                                      lambda: tf.image.rot90(memory_crop, -pose[2]))

                            return memory_crop_rot

                    with tf.variable_scope('map_crop'):
                        memory_crops_rot_batch = []
                        for bb in range(self.batch_size):
                            memory_crops_rot = tf.map_fn(
                                lambda elems: spatial_crop(*elems),
                                elems=(self.map_mask_placeholder[bb,...], self.pose_placeholder[bb,...]),
                                dtype=tf.float32)
                            memory_crops_rot_batch.append(memory_crops_rot)

                self.memory_crops_rot = tf.concat(memory_crops_rot_batch, axis=0)

                # Action decision network
                if constants.QUESTION_IN_ACTION:
                    question_embedding_tiled = tf.tile(question_embedding[:, tf.newaxis, tf.newaxis, tf.newaxis, :],
                                                       tf.stack((1, self.num_unrolls, self.crop_size, self.crop_size, 1)))
                    net = tf.concat((self.memory_crops_rot, tf_util.remove_axis(question_embedding_tiled, 1)), axis=3)
                else:
                    net = self.memory_crops_rot

                with tf.variable_scope('action_decision'):

                    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.elu):
                        net = slim.conv2d(net, 32, (1, 1), stride=1, padding='VALID', scope='conv_project')
                        net = slim.conv2d(net, 32, (3, 3), stride=1, padding='VALID', scope='conv1')
                        net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool1')
                        net = slim.conv2d(net, 64, (3, 3), stride=1, padding='VALID', scope='conv2')
                        net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool2')
                        net = slim.conv2d(net, 64, (3, 3), stride=1, padding='VALID', scope='conv3')
                        net = slim.conv2d(net, 32, (1, 1), stride=1, padding='VALID', scope='conv4')
                        net = slim.conv2d(net, 64, (3, 3), stride=1, padding='VALID', scope='conv5')
                        net = tf_util.remove_axis(net, axis=(2, 3))
                        net = tf.reshape(net, tf.stack((self.batch_size, self.num_unrolls, 1, net.get_shape().as_list()[-1])))
                        net = tf_util.remove_axis(net, axis=(1, 3))

                    net_point = net
                    net_point = tf.concat((net_point, self.gru_out), axis=1)
                    net_point = slim.fully_connected(net_point, 256, activation_fn=tf.nn.elu, scope='fc1')
                    net_point = slim.fully_connected(net_point, 256, activation_fn=tf.nn.elu, scope='fc2')
                    with tf.variable_scope('fc_action'):
                        teleport_weights = tf.clip_by_value(slim.fully_connected(net_point, constants.STEPS_AHEAD ** 2,
                                                                                 scope='teleport_action', activation_fn=None), -20, 20)
                        # rotate-left, rotate-right, look-up, look-down, open-obj, close-obj
                        action_weights = tf.clip_by_value(slim.fully_connected(net_point, 6, scope='interaction_action', activation_fn=None), -20, 20)
                        value = slim.fully_connected(net_point, 1, scope='value', activation_fn=None)

                with tf.variable_scope('action_prob'):
                    crop_mid = int(self.crop_size / 2)

                    # Teleport probs
                    input_crops = self.memory_crops_rot[:, crop_mid:crop_mid + constants.STEPS_AHEAD + 2,
                                  crop_mid - int(constants.STEPS_AHEAD / 2) - 1:crop_mid + int(constants.STEPS_AHEAD / 2 + 2), :]
                    self.teleport_input_crops = input_crops
                    net = slim.conv2d(input_crops, 1, (3, 3), stride=1, padding='VALID', scope='conv_teleport', activation_fn=None)
                    teleport_weights_success = tf_util.remove_axis(net, axis=(2, 3))

                    # Other actions probs
                    net = slim.conv2d(input_crops, 64, (3, 3), stride=1, padding='VALID', scope='conv_interaction', activation_fn=tf.nn.elu)
                    net_point = tf_util.remove_axis(net, axis=(2, 3))
                    net_point = tf.concat((net_point, self.gru_out), axis=1)
                    net_point = slim.fully_connected(net_point, 128, activation_fn=tf.nn.elu, scope='fc_success')
                    # rotate-left, rotate-right, look-up, look-down, open-obj, close-obj
                    action_weights_success = slim.fully_connected(net_point, 6, scope='interaction_success', activation_fn=None)

            # Question stuff
            with tf.variable_scope('question_answer'):
                pos_grid = (np.mgrid[0:constants.SPATIAL_MAP_HEIGHT, 0:constants.SPATIAL_MAP_WIDTH]).astype(np.float32)
                pos_grid -= np.mean(pos_grid)
                pos_grid /= pos_grid.max()
                pos_grid = tf.convert_to_tensor(pos_grid.transpose(1, 2, 0))
                pos_grid = tf.tile(pos_grid[tf.newaxis, tf.newaxis,...], tf.stack((self.batch_size, self.num_unrolls, 1, 1, 1)))
                memory = self.map_mask_placeholder
                memory = tf.concat((pos_grid, memory), axis=4)
                question_embedding = tf.tile(question_embedding[:, tf.newaxis, tf.newaxis, tf.newaxis, :],
                                             tf.stack((1, self.num_unrolls, constants.SPATIAL_MAP_HEIGHT, constants.SPATIAL_MAP_WIDTH, 1)))
                memory = tf_util.remove_axis(tf.concat((memory, question_embedding), axis=4), 1)

                with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=tf.nn.elu):
                    memory = slim.conv2d(memory, 64, (1, 1), stride=1, padding='VALID', scope='conv1_a')
                    memory = slim.conv2d(memory, 64, (1, 1), stride=1, padding='VALID', scope='conv1_b')
                    memory = slim.max_pool2d(memory, [2, 2], stride=2, padding='VALID', scope='pool1')
                    memory = slim.conv2d(memory, 128, (3, 3), stride=1, padding='VALID', scope='conv2')
                    memory = slim.max_pool2d(memory, [2, 2], stride=2, padding='VALID', scope='pool2')
                    memory = slim.conv2d(memory, 128, (3, 3), stride=1, padding='VALID', scope='conv3')
                    memory = slim.max_pool2d(memory, [2, 2], stride=2, padding='VALID', scope='pool3')
                    memory = slim.conv2d(memory, 128, (3, 3), stride=1, padding='VALID', scope='conv4')
                    memory = tf.reduce_mean(memory, axis=(1, 2))
                    memory = tf.concat((memory,
                                        tf.reshape(self.episode_length_placeholder, (-1, 1)),
                                        tf.reshape(self.question_count_placeholder, (-1, 1))),
                                       axis=1)
                    memory = slim.fully_connected(memory, 128, scope='fc1')
                    memory = slim.fully_connected(memory, 128, scope='fc2')

                answer_weight = tf.clip_by_value(slim.fully_connected(memory, 1, scope='answer_weight', activation_fn=None), -20, 20)
                answer_value = slim.fully_connected(memory, 1, scope='answer_v', activation_fn=None)

                existence_answer = slim.fully_connected(memory, 1, scope='existence_answer', activation_fn=None)
                self.existence_answer_logits = tf.reshape(existence_answer, (self.batch_size, -1))
                self.existence_answer = tf.nn.sigmoid(self.existence_answer_logits)

                counting_answer = slim.fully_connected(memory, constants.MAX_COUNTING_ANSWER + 1, scope='counting_answer', activation_fn=None)
                self.counting_answer_logits = tf.reshape(counting_answer, (self.batch_size, -1, constants.MAX_COUNTING_ANSWER + 1))
                self.counting_answer = tf.nn.softmax(counting_answer)

            with tf.variable_scope('a3c_pi_v'):
                self.v = (value + answer_value)[:, 0]
                self.possible_moves_logits = tf.concat((teleport_weights_success, action_weights_success), axis=1)
                self.possible_moves = tf.stop_gradient(tf.nn.sigmoid(self.possible_moves_logits))
                self.possible_moves_logits = tf.reshape(self.possible_moves_logits, tf.stack((self.batch_size, self.num_unrolls,
                                                                                              self.possible_moves_logits.get_shape().as_list()[-1])))
                self.possible_moves_weights = tf.concat((self.possible_moves, tf.ones(tf.stack((self.batch_size * self.num_unrolls, 1)))), axis=1)
                self.actions = tf.concat((teleport_weights, action_weights, answer_weight), axis=1)
                self.pi_logits = self.actions
                if constants.USE_POSSIBLE_PRIOR:
                    self.pi = tf.exp(self.actions) * tf.round(self.possible_moves_weights)
                    self.pi /= tf.maximum(tf.reduce_sum(self.pi, axis=1)[:, tf.newaxis], 1e-10)
                else:
                    self.pi = tf.nn.softmax(self.actions)

    def answer_loss(self):
        with tf.variable_scope('answer_loss'):
            self.answer_weight = tf.placeholder(tf.float32, [self.batch_size, None], name='answer_weight')

            self.existence_answer_placeholder = tf.placeholder(tf.float32, [self.batch_size, None], name='existence_answer_placeholder')
            existence_answer_loss = tf.losses.sigmoid_cross_entropy(
                multi_class_labels=self.existence_answer_placeholder,
                logits=self.existence_answer_logits,
                weights=self.answer_weight * tf.cast(tf.not_equal(self.question_type_placeholder, 1), tf.float32),
            )

            self.counting_answer_placeholder = tf.placeholder(tf.int32, [self.batch_size, None], name='counting_answer_placeholder')
            counting_answer_loss = tf.losses.softmax_cross_entropy(
                onehot_labels=tf.one_hot(self.counting_answer_placeholder, constants.MAX_COUNTING_ANSWER + 1),
                logits=self.counting_answer_logits,
                weights=self.answer_weight * tf.cast(tf.equal(self.question_type_placeholder, 1), tf.float32),
            )

            answer_loss = existence_answer_loss + counting_answer_loss

            if constants.SUPERVISED:
                self.rl_total_loss = answer_loss
            else:
                self.rl_total_loss += answer_loss

    def move_loss(self):
        with tf.variable_scope('move_loss'):
            move_loss = 10 * tf.losses.sigmoid_cross_entropy(
                multi_class_labels=self.possible_move_placeholder,
                logits=self.possible_moves_logits)

            if constants.USE_POSSIBLE_PRIOR:
                self.rl_total_loss += move_loss
