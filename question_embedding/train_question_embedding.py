import numpy as np
import tensorflow as tf
import glob
import time
import os

from utils import py_util
from utils import tf_util

import constants
from question_embedding import parse_question
from networks.question_embedding_network import QuestionEmbeddingNetwork


def shuffle_data(data):
    # shuffle the data
    num_data_points = -1
    new_data = {}
    for key, val in data.items():
        if num_data_points == -1:
            num_data_points = val.shape[0]
            random_order = np.random.permutation(np.arange(num_data_points))
        new_data[key] = val[random_order].copy()  # makes permuted data contiguous
    return new_data


def run():
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(constants.GPU_ID)
        sess = tf_util.Session()
        network = QuestionEmbeddingNetwork(sess, constants.BATCH_SIZE)
        network.create_network()
        network.create_train_ops()

        if not constants.DEBUG:
            log_writer = tf.summary.FileWriter(os.path.join(constants.LOG_FILE, py_util.get_time_str()), sess.graph)
            with tf.name_scope('train'):
                train_summary = tf.summary.merge([
                    tf.summary.scalar('total_loss', network.loss),
                    tf.summary.scalar('question_type_loss', network.question_type_loss),
                    tf.summary.scalar('object_id_loss', network.object_id_loss),
                    tf.summary.scalar('container_id_loss', network.container_id_loss),
                    tf.summary.scalar('question_type_accuracy', network.question_type_accuracy),
                    tf.summary.scalar('object_id_accuracy', network.object_id_accuracy),
                    tf.summary.scalar('container_id_accuracy', network.container_id_accuracy),
                    ])

            with tf.name_scope('test'):
                test_summary_ph = tf.placeholder(tf.float32, [None], 'test_summary_placeholder')
                test_summary = tf.summary.merge([
                    tf.summary.scalar('total_loss', test_summary_ph[0]),
                    tf.summary.scalar('question_type_loss', test_summary_ph[1]),
                    tf.summary.scalar('object_id_loss', test_summary_ph[2]),
                    tf.summary.scalar('container_id_loss', test_summary_ph[3]),
                    tf.summary.scalar('question_type_accuracy', test_summary_ph[4]),
                    tf.summary.scalar('object_id_accuracy', test_summary_ph[5]),
                    tf.summary.scalar('container_id_accuracy', test_summary_ph[6]),
                ])

        data_train = sorted(glob.glob(os.path.join('questions', 'train', '*', '*.csv')))
        data_train = [parse_question.get_sequences(filename) for filename in data_train]
        data_train_dict = {}
        for data in data_train:
            for key, val in data.items():
                current_data = data_train_dict.get(key, [])
                current_data.extend(val)
                data_train_dict[key] = current_data
        data_train = data_train_dict
        data_train = {key: np.array(val) for key, val in data_train.items()}
        num_data_train = data_train['questions'].shape[0]

        data_test = sorted(glob.glob(os.path.join('questions', 'train_test', '*', '*.csv')))
        data_test = [parse_question.get_sequences(filename) for filename in data_test]
        data_test_dict = {}
        for data in data_test:
            for key, val in data.items():
                current_data = data_test_dict.get(key, [])
                current_data.extend(val)
                data_test_dict[key] = current_data
        data_test = data_test_dict
        data_test = {key: np.array(val) for key, val in data_test.items()}
        num_data_test = data_test['questions'].shape[0]

        sess.run(tf.global_variables_initializer())
        iteration = network.restore()
        sess.graph.finalize()
        num_its = 0

        data_train = shuffle_data(data_train)
        data_ind = 0
        total_iteration_time = 0
        while iteration < constants.MAX_TIME_STEP:
            t_start = time.time()
            if num_its == 0 or data_ind + constants.BATCH_SIZE > num_data_train:
                # end of epoch
                data_ind = 0
                data_train = shuffle_data(data_train)

                if not constants.DEBUG:
                    # run test
                    test_start = time.time()
                    test_losses = []
                    for test_data_ind in range(0, num_data_test - constants.BATCH_SIZE + 1, constants.BATCH_SIZE):
                        data_batch = {key: val[test_data_ind:test_data_ind + constants.BATCH_SIZE, ...] for key, val in data_test.items()}
                        feed_dict = {
                            network.question_placeholder: data_batch['questions'],
                            network.question_type_label: data_batch['question_types'],
                            network.object_id_label: data_batch['object_ids'],
                            network.container_id_label: data_batch['container_ids'],
                            }
                        losses = sess.run([network.loss,
                                           network.question_type_loss, network.object_id_loss, network.container_id_loss,
                                           network.question_type_accuracy, network.object_id_accuracy, network.container_id_accuracy
                                           ], feed_dict=feed_dict)
                        test_losses.append(losses)
                    summary_losses = np.mean(test_losses, axis=0)
                    summary = sess.run(test_summary, feed_dict={test_summary_ph: summary_losses})
                    log_writer.add_summary(summary, iteration)
                    log_writer.flush()
                    print('test time               %.3f' % (time.time() - test_start))

            data_batch = {key: val[data_ind:data_ind + constants.BATCH_SIZE, ...] for key, val in data_train.items()}
            feed_dict = {
                network.question_placeholder: data_batch['questions'],
                network.question_type_label: data_batch['question_types'],
                network.object_id_label: data_batch['object_ids'],
                network.container_id_label: data_batch['container_ids'],
                }
            if num_its % 10 == 0 and not constants.DEBUG:
                _, summary = sess.run([network.train_op, train_summary], feed_dict=feed_dict)
                log_writer.add_summary(summary, iteration)
                log_writer.flush()
            else:
                sess.run(network.train_op, feed_dict=feed_dict)
            data_ind += constants.BATCH_SIZE

            t_end = time.time()
            iteration_time = t_end - t_start
            total_iteration_time += iteration_time
            if num_its % 100 == 0:
                print('\niteration               %d' % iteration)
                print('time per iteration      %.3f' % (total_iteration_time / (num_its + 1)))
                print('time for last iteration %.3f' % (iteration_time))
            num_its += 1
            iteration += 1
            if iteration % 1000 == 0:
                network.save(iteration)
    except KeyboardInterrupt:
        pass
    except:
        import traceback
        traceback.print_exc()
        print('error')
    finally:
        if not constants.DEBUG:
            print('\nsaving')
            network.save(iteration)


if __name__ == '__main__':
    run()
