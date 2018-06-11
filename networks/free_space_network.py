import pdb
import tensorflow as tf
from utils import tf_util
from graph import batchGraphGRU
from graph import graph_obj

import constants

slim = tf.contrib.slim


class FreeSpaceNetwork(object):
    def __init__(self, gru_size, batch_size, num_unrolls=None):
        self.gru_size = gru_size
        self.batch_size = batch_size
        self.num_unrolls = num_unrolls
        if num_unrolls is None:
            self.num_unrolls = tf.placeholder(tf.int32, name='num_unrolls')

        self.image_placeholder = tf.placeholder(tf.uint8, [batch_size, num_unrolls, constants.SCREEN_HEIGHT, constants.SCREEN_WIDTH, 3], name='image_placeholder')
        self.memory_placeholders = tf.placeholder(tf.float32, [batch_size, constants.SPATIAL_MAP_HEIGHT, constants.SPATIAL_MAP_WIDTH, constants.MEMORY_SIZE], name='memory_placeholder')
        self.labels_placeholder = tf.placeholder(tf.float32, [batch_size, num_unrolls, constants.STEPS_AHEAD, constants.STEPS_AHEAD , 1], name='labels_placeholder')
        self.action_placeholder = tf.placeholder(tf.float32, [batch_size, num_unrolls, 3], name='action_placeholder') # move ahead, rotate left, rotate right
        self.gru_placeholder = tf.placeholder(tf.float32, [batch_size, gru_size], name='gru_placeholder')
        self.mask_placeholder = tf.placeholder(tf.float32, [batch_size, num_unrolls], name='mask_placeholder')
        self.pose_placeholder = tf.placeholder(tf.int32, [batch_size, num_unrolls, 3], name='pose_placeholder')
        self.pose_indicator_placeholder = tf.placeholder(tf.float32,
                [batch_size, num_unrolls, constants.TERMINAL_CHECK_PADDING * 2 + 1, constants.TERMINAL_CHECK_PADDING * 2 + 1],
                name='pose_indicator_placeholder')

    def create_net(self, add_loss=True):
        self.conv_layers()
        self.fc_layers()
        self.gru_layer()
        self.gru_outputs_full, self.gru_output_patches = self.map_layers()
        self.gru_outputs = self.gru_outputs_full[:,-1,...]
        self.output_layers()
        if add_loss:
            self.loss = self.supervised_loss()
            self.training_op = self.training(self.loss)

    def conv_layers(self):
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.elu):
            inputs = (tf.to_float(tf_util.remove_axis(self.image_placeholder, 1)) - 128) / 128
            net = slim.conv2d(inputs, 64, [7, 7], stride=2, scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.conv2d(net, 128, [5, 5], stride=1, scope='conv2')
            net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool2')
            net = slim.conv2d(net, 256, [3, 3], stride=1, scope='conv3_a')
            net = slim.conv2d(net, 256, [3, 3], stride=1, scope='conv3_b')
            net = slim.conv2d(net, 256, [3, 3], stride=1, scope='conv3_c')
            net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool3')
            net = slim.conv2d(net, 512, [3, 3], stride=1, scope='conv4_a')
            net = slim.conv2d(net, 512, [3, 3], stride=1, scope='conv4_b')
            net = slim.conv2d(net, 512, [3, 3], stride=1, scope='conv4_c')
            net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool4')
        self.conv_output = net

    def fc_layers(self):
        net = self.conv_output
        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.elu):
            net = tf_util.remove_axis(net, [2, 3])
            net = slim.fully_connected(net, 1024, scope='fc1')
            action = slim.fully_connected(tf_util.remove_axis(self.action_placeholder, 1), 32, scope='fc_action')
            net = tf.concat((net, action), axis=1)
        self.fc_output = net

    def gru_layer(self):
        # FC GRU
        net = self.fc_output
        gru = tf.contrib.rnn.GRUCell(self.gru_size)
        net = tf.reshape(net, tf.stack((self.batch_size, self.num_unrolls, net.get_shape().as_list()[1])))
        self.fc_gru_outputs, self.gru_state = tf.nn.dynamic_rnn(
                gru, net, initial_state=self.gru_placeholder, swap_memory=False)
        net = tf.concat((net, self.fc_gru_outputs), axis=2)
        self.gru_out = tf_util.remove_axis(net, 1)
        net = slim.fully_connected(net,  1024, scope='fc2')
        net = tf.reshape(net, (self.batch_size, self.num_unrolls, net.get_shape().as_list()[2]))
        self.gru_output = net

    def map_layers(self):
        with tf.variable_scope('gru'):
            gru = batchGraphGRU.BatchGraphGRUCell()
            net = tf.reshape(self.gru_output, tf.stack((self.batch_size, self.num_unrolls, -1)))
            gru_outputs_tuple, final_state = tf.nn.dynamic_rnn(
                    gru, (net, self.pose_placeholder),
                    initial_state=(tf.reshape(self.memory_placeholders, [self.batch_size, -1]),
                        tf.zeros((self.batch_size, constants.MEMORY_SIZE * constants.STEPS_AHEAD * constants.STEPS_AHEAD))),
                    swap_memory=False)
            gru_outputs = tf.reshape(gru_outputs_tuple[0],
                    (self.batch_size, self.num_unrolls, constants.SPATIAL_MAP_HEIGHT, constants.SPATIAL_MAP_WIDTH, constants.MEMORY_SIZE))
            gru_output_patches = tf.reshape(gru_outputs_tuple[1],
                    (self.batch_size, self.num_unrolls, constants.STEPS_AHEAD, constants.STEPS_AHEAD, constants.MEMORY_SIZE))
        return gru_outputs, gru_output_patches

    def output_layers(self):
        with tf.variable_scope('conv_output'):
            patch_weights = slim.conv2d(
                    tf_util.remove_axis(self.gru_output_patches, 1),
                    1, (1, 1), stride=1, activation_fn=None, scope='conv_output')
            patch_weights = tf_util.split_axis(patch_weights, 0, self.batch_size, self.num_unrolls)
            patch_weights_sigm = tf.nn.sigmoid(patch_weights)
            patch_weights_clipped = tf.minimum(tf.maximum(1.0 + graph_obj.EPSILON, 5 * tf.exp(patch_weights)), graph_obj.MAX_WEIGHT)

        with tf.variable_scope('conv_output', reuse=True):
            map_weights = slim.conv2d(
                    tf_util.remove_axis(self.gru_outputs_full, 1),
                    1, (1, 1), stride=1, activation_fn=None, scope='conv_output')
            map_weights_clipped = tf.minimum(tf.maximum(1.0 + graph_obj.EPSILON, 5 * tf.exp(map_weights)), graph_obj.MAX_WEIGHT)
            self.occupancy = tf.nn.sigmoid(map_weights)

        self.patch_weights = patch_weights
        self.patch_weights_sigm = patch_weights_sigm
        self.patch_weights_clipped = patch_weights_clipped
        self.map_weights_clipped = map_weights_clipped

        with tf.variable_scope('terminal'):
            self.goal_pose_placeholder = tf.placeholder(tf.int32, [self.batch_size, 2], name='goal_pose_placeholder')
            terminal_patches = []
            for bb in range(self.batch_size):
                terminal_patch = self.gru_outputs_full[bb][:,
                        self.goal_pose_placeholder[bb, 1] - constants.TERMINAL_CHECK_PADDING:self.goal_pose_placeholder[bb, 1] + constants.TERMINAL_CHECK_PADDING + 1,
                        self.goal_pose_placeholder[bb, 0] - constants.TERMINAL_CHECK_PADDING:self.goal_pose_placeholder[bb, 0] + constants.TERMINAL_CHECK_PADDING + 1, :]
                terminal_patch = tf.reshape(terminal_patch,
                        tf.stack((self.num_unrolls, constants.TERMINAL_CHECK_PADDING * 2 + 1, constants.TERMINAL_CHECK_PADDING * 2 + 1, constants.MEMORY_SIZE)))
                terminal_patch = tf.concat((terminal_patch, self.pose_indicator_placeholder[bb,..., tf.newaxis]), axis=3)
                terminal_patches.append(terminal_patch)
            terminal_patches = tf.stack(terminal_patches, axis=0)
            self.terminal_patches = terminal_patches
            terminal_patches = slim.conv2d(tf_util.remove_axis(terminal_patches, 1), 32, [3, 3], stride=1, scope='conv1', activation_fn=tf.nn.elu)
            terminal_patches = slim.conv2d(terminal_patches, 32, [3, 3], stride=1, scope='conv2', activation_fn=tf.nn.elu)
            self.is_possible = tf.reshape(slim.fully_connected(
                tf_util.remove_axis(terminal_patches, axis=(2, 3)), 1, scope='fc_possible', activation_fn=None),
                tf.stack((self.batch_size, self.num_unrolls)))
            self.is_possible_sigm = tf.nn.sigmoid(self.is_possible)

    def supervised_loss(self):
        total_loss = 0
        loss = 10 * tf.losses.sigmoid_cross_entropy(
                multi_class_labels=tf_util.remove_axis(self.labels_placeholder, axis=(1, 3, 4)),
                logits=tf_util.remove_axis(self.patch_weights, axis=(1, 3, 4)),
                weights=tf_util.remove_axis(self.mask_placeholder, 1)[:, tf.newaxis])
        total_loss += loss

        self.possible_label_placeholder = tf.placeholder(tf.float32, [self.batch_size, self.num_unrolls], name='possible_label_placeholder')
        possible_loss = tf.losses.sigmoid_cross_entropy(
                multi_class_labels=tf.reshape(self.possible_label_placeholder, (-1, 1)),
                logits=tf.reshape(self.is_possible, (-1, 1)),
                weights=tf.reshape(self.mask_placeholder, (-1, 1)))
        total_loss += possible_loss
        return total_loss

    def regularizer_layers(self):
        return tf_util.l2_regularizer()

    def training(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        train_op = slim.learning.create_train_op(loss, optimizer)
        return train_op
