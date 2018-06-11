import tensorflow as tf
from utils import tf_util
from networks.rl_network import A3CNetwork

import constants

slim = tf.contrib.slim


class EndToEndBaselineNetwork(A3CNetwork):
    def __init__(self):
        self.num_unrolls = tf.placeholder(tf.int32, name='num_unrolls')

        self.image_placeholder = tf.placeholder(tf.uint8, [1, None, constants.SCREEN_HEIGHT, constants.SCREEN_WIDTH, 3], name='image_placeholder')

        # Move ahead, Rotate Left/Right, Look Up/Down, Open/Close, Answer
        self.action_placeholder = tf.placeholder(tf.float32, [1, None, 7], name='action_placeholder')
        self.gru_placeholder = tf.placeholder(tf.float32, [1, 64], name='gru_placeholder')
        self.detection_image = tf.placeholder(tf.float32, [1, None, constants.SCREEN_HEIGHT, constants.SCREEN_WIDTH, constants.NUM_CLASSES], name='detection_image')
        self.answer_weight = tf.placeholder(tf.float32, [1, None], name='answer_weight')

    def create_net(self):
        self.conv_layers()
        self.fc_layers()
        self.gru_layer()
        self.rl_layers()
        self.rl_loss()
        self.answer_loss()

    def conv_layers(self):
        inputs = (tf.to_float(tf_util.remove_axis(self.image_placeholder, 1)) - 128) / 128.0
        obj_dets = (tf_util.remove_axis(self.detection_image, 1) - 128) / 128.0
        if constants.USE_OBJECT_DETECTION_AS_INPUT:
            inputs = tf.concat((inputs, obj_dets), axis=3)
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.elu):
            net = slim.conv2d(inputs, 32, [7, 7], stride=2, scope='conv1')
            net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool1')
            net = slim.conv2d(net, 64, [5, 5], stride=1, scope='conv2')
            net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool2')
            net = slim.conv2d(net, 128, [3, 3], stride=1, scope='conv3')
            net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool3')
            net = slim.conv2d(net, 256, [3, 3], stride=1, scope='conv4')
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
        net = self.fc_output
        gru = tf.contrib.rnn.GRUCell(constants.RL_GRU_SIZE)
        net = tf.reshape(net, tf.stack((1, self.num_unrolls, net.get_shape().as_list()[1])))
        self.fc_gru_outputs, self.gru_state = tf.nn.dynamic_rnn(
                gru, net, initial_state=self.gru_placeholder, swap_memory=False)
        net = tf.concat((net, self.fc_gru_outputs), axis=2)
        self.gru_out = tf_util.remove_axis(net, 1)

    def rl_layers(self):
        inputs = slim.fully_connected(self.gru_out, 256, scope='rl_fc1', activation_fn=tf.nn.elu)

        with tf.variable_scope('question_answer'):
            self.question_object_placeholder = tf.placeholder(tf.int32, [1, None], name='question_object_placeholder')
            self.question_object_one_hot = tf.one_hot(self.question_object_placeholder[:, 0], constants.NUM_CLASSES)

            self.question_container_placeholder = tf.placeholder(tf.float32, [1, None, constants.NUM_CLASSES], name='question_container_placeholder')
            self.question_container_one_hot = self.question_container_placeholder[:, 0, :]

            self.question_direction_placeholder = tf.placeholder(tf.float32, [1, None, 4], name='question_direction_placeholder')
            self.question_direction_one_hot = self.question_direction_placeholder[:, 0, :]

            self.question_type_placeholder = tf.placeholder(tf.int32, [1, None], name='question_type_placeholder')
            self.question_type_one_hot = tf.one_hot(self.question_type_placeholder[:, 0], 4)

            with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.elu):
                question_embedding = slim.fully_connected(self.question_object_one_hot, 64, scope='question')
                question_container_embedding = slim.fully_connected(self.question_container_one_hot, 64, scope='question_container')
                question_direction_embedding = slim.fully_connected(self.question_direction_one_hot, 8, scope='question_direction')
                question_type_embedding = slim.fully_connected(self.question_type_one_hot, 8, scope='question_type')
                question_embedding = tf.concat((question_embedding, question_type_embedding, question_container_embedding, question_direction_embedding), axis=1)
                question_embedding = tf.tile(question_embedding, tf.stack((self.num_unrolls, 1)))

        inputs = tf.concat((inputs, question_embedding), axis=1)
        inputs = slim.fully_connected(inputs, 256, scope='rl_fc2', activation_fn=tf.nn.elu)
        inputs = slim.fully_connected(inputs, 256, scope='rl_fc3', activation_fn=tf.nn.elu)

        self.pi = slim.fully_connected(inputs, 8, scope='fc_policy', activation_fn=tf.nn.softmax)
        self.v = slim.fully_connected(inputs, 1, scope='fc_value', activation_fn=None)[:, 0]

        answer_weight = slim.fully_connected(inputs, 1, scope='existence_answer', activation_fn=None)
        self.existence_answer_logits = answer_weight[tf.newaxis, :, 0]
        self.existence_answer = tf.nn.sigmoid(self.existence_answer_logits)

        answer_weight = slim.fully_connected(inputs, constants.MAX_COUNTING_ANSWER + 1, scope='counting_answer', activation_fn=None)
        self.counting_answer_logits = answer_weight[tf.newaxis,...]
        self.counting_answer = tf.nn.softmax(answer_weight)

    def answer_loss(self):
        self.existence_answer_placeholder = tf.placeholder(tf.float32, [1, None], name='existence_answer_placeholder')
        existence_answer_loss = tf.losses.sigmoid_cross_entropy(
            multi_class_labels=self.existence_answer_placeholder,
            logits=self.existence_answer_logits,
            weights=self.answer_weight * tf.cast(tf.not_equal(self.question_type_placeholder, 1), tf.float32),
        )

        self.counting_answer_placeholder = tf.placeholder(tf.int32, [1, None], name='counting_answer_placeholder')
        counting_answer_loss = tf.losses.softmax_cross_entropy(
            onehot_labels=tf.one_hot(self.counting_answer_placeholder, constants.MAX_COUNTING_ANSWER + 1),
            logits=self.counting_answer_logits,
            weights=self.answer_weight * tf.cast(tf.equal(self.question_type_placeholder, 1), tf.float32),
        )

        answer_loss = existence_answer_loss + counting_answer_loss

        self.rl_total_loss += answer_loss