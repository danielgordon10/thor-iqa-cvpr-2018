import pdb
import os
import tensorflow as tf

from question_embedding import parse_question
import constants
from utils import tf_util


class QuestionEmbeddingNetwork(object):
    def __init__(self, sess, batch_size=None, input_length=constants.MAX_SENTENCE_LENGTH, model_name='question_embedding_network'):
        self.sess = sess
        self.batch_size = batch_size
        self.input_length = input_length
        self.model_name = model_name

    def create_network(self):
        embedding_size = 32
        with tf.variable_scope(self.model_name):
            self.question_placeholder = tf.placeholder(tf.int32, [self.batch_size, None], name='question_placeholder')
            question_one_hot = tf.one_hot(self.question_placeholder, parse_question.vocabulary_size())
            reshape_one_hot = tf_util.remove_axis(question_one_hot, axis=1)
            fc1 = tf.contrib.layers.fully_connected(reshape_one_hot, embedding_size, activation_fn=tf.nn.elu)
            unsqueeze_fc1 = tf_util.split_axis(fc1, 0, self.batch_size, -1)
            lstm = tf.contrib.rnn.BasicLSTMCell(embedding_size)
            initial_state = lstm.zero_state(self.batch_size, tf.float32)
            outputs, state = tf.nn.dynamic_rnn(lstm, unsqueeze_fc1, None, initial_state)
            self.question_embedding = tf.contrib.layers.fully_connected(state.h, embedding_size, activation_fn=tf.nn.elu)
            self.question_type_logits = tf.contrib.layers.fully_connected(self.question_embedding, len(constants.USED_QUESTION_TYPES), activation_fn=None)
            self.object_id_logits = tf.contrib.layers.fully_connected(self.question_embedding, len(constants.OBJECTS), activation_fn=None)
            self.container_id_logits = tf.contrib.layers.fully_connected(self.question_embedding, len(constants.OBJECTS) + 1, activation_fn=None)
            self.question_type_pred = tf.argmax(self.question_type_logits, axis=1, output_type=tf.int32)
            self.object_id_pred = tf.argmax(self.object_id_logits, axis=1, output_type=tf.int32)
            self.container_id_pred = tf.argmax(self.container_id_logits, axis=1, output_type=tf.int32)

        self.saver = tf.train.Saver(tf.global_variables(self.model_name), max_to_keep=1)

    def create_train_ops(self):
        with tf.variable_scope(self.model_name):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=constants.LEARNING_RATE)
            self.question_type_label = tf.placeholder(tf.int32, [self.batch_size], name='question_type_label_placeholder')
            self.object_id_label = tf.placeholder(tf.int32, [self.batch_size], name='object_id_label_placeholder')
            self.container_id_label = tf.placeholder(tf.int32, [self.batch_size], name='container_id_label_placeholder')
            self.question_type_loss = tf.losses.sparse_softmax_cross_entropy(self.question_type_label, self.question_type_logits)
            self.object_id_loss = tf.losses.sparse_softmax_cross_entropy(self.object_id_label, self.object_id_logits)
            self.container_id_loss = tf.losses.sparse_softmax_cross_entropy(self.container_id_label, self.container_id_logits)

            self.question_type_accuracy = tf.reduce_mean(tf.cast(tf.equal(self.question_type_label, self.question_type_pred), tf.float32))
            self.object_id_accuracy = tf.reduce_mean(tf.cast(tf.equal(self.object_id_label, self.object_id_pred), tf.float32))
            self.container_id_accuracy = tf.reduce_mean(tf.cast(tf.equal(self.container_id_label, self.container_id_pred), tf.float32))

            self.loss = self.question_type_loss + self.object_id_loss + self.container_id_loss
            self.train_op = self.optimizer.minimize(self.loss)

    def restore(self):
        path = os.path.join(constants.CHECKPOINT_DIR)
        print('loading ', path)
        iteration = tf_util.restore_from_dir(self.sess, path)
        return iteration

    def save(self, iteration):
        tf_util.save(self.saver, self.sess, constants.CHECKPOINT_DIR, iteration)

    def forward_str(self, question_str):
        tokenized_question = parse_question.tokenize_sentence(question_str)
        return self.forward(tokenized_question)

    def forward(self, question):
        return self.sess.run(self.question_embedding, feed_dict={self.question_placeholder: question})

    def predict_str(self, question_str):
        tokenized_question = parse_question.tokenize_sentence(question_str)
        return self.predict(tokenized_question)

    def predict(self, question):
        return self.sess.run([self.question_type_pred, self.object_id_pred, self.container_id_pred],
                feed_dict={self.question_placeholder: question})


