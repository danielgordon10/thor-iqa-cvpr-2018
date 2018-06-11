# -*- coding: utf-8 -*-
import pdb
import tensorflow as tf
import threading
import numpy as np

import random
import os
import time

from networks.qa_planner_network import QAPlannerNetwork
from networks.free_space_network import FreeSpaceNetwork
from networks.end_to_end_baseline_network import EndToEndBaselineNetwork
from reinforcement_learning.a3c_training_thread import A3CTrainingThread
from reinforcement_learning.a3c_testing_thread import A3CTestingThread
from reinforcement_learning.rmsprop_applier import RMSPropApplier
from utils import tf_util
from utils import py_util

import constants

global_t = 0
dataset = None
data_ind = 0


def run():
    global global_t
    global dataset
    global data_ind
    if constants.OBJECT_DETECTION:
        from darknet_object_detection import detector
        detector.setup_detectors(constants.PARALLEL_SIZE)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(constants.GPU_ID)

    try:
        with tf.variable_scope('global_network'):
            if constants.END_TO_END_BASELINE:
                global_network = EndToEndBaselineNetwork()
            else:
                global_network = QAPlannerNetwork(constants.RL_GRU_SIZE, 1, 1)
            global_network.create_net()
        if constants.USE_NAVIGATION_AGENT:
            with tf.variable_scope('nav_global_network') as net_scope:
                free_space_network = FreeSpaceNetwork(constants.GRU_SIZE, 1, 1)
                free_space_network.create_net()
        else:
            net_scope = None

        conv_var_list = [v for v in tf.trainable_variables() if 'conv' in v.name and 'weight' in v.name and
                        (v.get_shape().as_list()[0] != 1 or v.get_shape().as_list()[1] != 1)]
        for var in conv_var_list:
            tf_util.conv_variable_summaries(var, scope=var.name.replace('/', '_')[:-2])
        conv_image_summary = tf.summary.merge_all()

        # prepare session
        sess = tf_util.Session()

        # Instantiate singletons without scope.
        if constants.PREDICT_DEPTH:
            from depth_estimation_network import depth_estimator
            with tf.variable_scope('') as depth_scope:
                depth_estimator = depth_estimator.get_depth_estimator(sess)
        else:
            depth_scope = None

        training_threads = []

        learning_rate_input = tf.placeholder(tf.float32, name='learning_rate')
        grad_applier = RMSPropApplier(learning_rate=learning_rate_input,
                decay=constants.RMSP_ALPHA,
                momentum=0.0,
                epsilon=constants.RMSP_EPSILON,
                clip_norm=constants.GRAD_NORM_CLIP)

        for i in range(constants.PARALLEL_SIZE):
            training_thread = A3CTrainingThread(
                    i, sess,
                    learning_rate_input,
                    grad_applier,
                    constants.MAX_TIME_STEP,
                    net_scope,
                    depth_scope)
            training_threads.append(training_thread)

        if constants.RUN_TEST:
            testing_thread = A3CTestingThread(constants.PARALLEL_SIZE + 1, sess, net_scope, depth_scope)

        sess.run(tf.global_variables_initializer())

        # Initialize pretrained weights after init.
        if constants.PREDICT_DEPTH:
            depth_estimator.load_weights()


        episode_reward_input = tf.placeholder(tf.float32, name='ep_reward')
        episode_length_input = tf.placeholder(tf.float32, name='ep_length')
        exist_answer_correct_input = tf.placeholder(tf.float32, name='exist_ans')
        count_answer_correct_input = tf.placeholder(tf.float32, name='count_ans')
        contains_answer_correct_input = tf.placeholder(tf.float32, name='contains_ans')
        percent_invalid_actions_input = tf.placeholder(tf.float32, name='inv')

        scalar_summaries = [
            tf.summary.scalar("Episode Reward", episode_reward_input),
            tf.summary.scalar("Episode Length", episode_length_input),
            tf.summary.scalar("Percent Invalid Actions", percent_invalid_actions_input),
        ]
        exist_summary = tf.summary.scalar("Answer Correct Existence", exist_answer_correct_input),
        count_summary = tf.summary.scalar("Answer Correct Counting", count_answer_correct_input),
        contains_summary = tf.summary.scalar("Answer Correct Containing", contains_answer_correct_input),
        exist_summary_op = tf.summary.merge(scalar_summaries + [exist_summary])
        count_summary_op = tf.summary.merge(scalar_summaries + [count_summary])
        contains_summary_op = tf.summary.merge(scalar_summaries + [contains_summary])
        summary_ops = [exist_summary_op, count_summary_op, contains_summary_op]

        summary_placeholders = {
            "episode_reward_input": episode_reward_input,
            "episode_length_input": episode_length_input,
            "exist_answer_correct_input": exist_answer_correct_input,
            "count_answer_correct_input": count_answer_correct_input,
            "contains_answer_correct_input": contains_answer_correct_input,
            "percent_invalid_actions_input" : percent_invalid_actions_input,
            }

        if constants.RUN_TEST:
            test_episode_reward_input = tf.placeholder(tf.float32, name='test_ep_reward')
            test_episode_length_input = tf.placeholder(tf.float32, name='test_ep_length')
            test_exist_answer_correct_input = tf.placeholder(tf.float32, name='test_exist_ans')
            test_count_answer_correct_input = tf.placeholder(tf.float32, name='test_count_ans')
            test_contains_answer_correct_input = tf.placeholder(tf.float32, name='test_contains_ans')
            test_percent_invalid_actions_input = tf.placeholder(tf.float32, name='test_inv')

            test_scalar_summaries = [
                tf.summary.scalar("Test Episode Reward", test_episode_reward_input),
                tf.summary.scalar("Test Episode Length", test_episode_length_input),
                tf.summary.scalar("Test Percent Invalid Actions", test_percent_invalid_actions_input),
            ]
            exist_summary = tf.summary.scalar("Test Answer Correct Existence", test_exist_answer_correct_input),
            count_summary = tf.summary.scalar("Test Answer Correct Counting", test_count_answer_correct_input),
            contains_summary = tf.summary.scalar("Test Answer Correct Containing", test_contains_answer_correct_input),
            test_exist_summary_op = tf.summary.merge(test_scalar_summaries + [exist_summary])
            test_count_summary_op = tf.summary.merge(test_scalar_summaries + [count_summary])
            test_contains_summary_op = tf.summary.merge(test_scalar_summaries + [contains_summary])
            test_summary_ops = [test_exist_summary_op, test_count_summary_op, test_contains_summary_op]

        if not constants.DEBUG:
            time_str = py_util.get_time_str()
            summary_writer = tf.summary.FileWriter(os.path.join(constants.LOG_FILE, time_str), sess.graph)
        else:
            summary_writer = None

        # init or load checkpoint with saver
        vars_to_save = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='global_network')
        saver = tf.train.Saver(vars_to_save, max_to_keep=3)
        print('-------------- Looking for checkpoints in ', constants.CHECKPOINT_DIR)
        global_t = tf_util.restore_from_dir(sess, constants.CHECKPOINT_DIR)

        if constants.USE_NAVIGATION_AGENT:
            print('now trying to restore nav model')
            tf_util.restore_from_dir(sess, 'logs/checkpoints/navigation', True)

        sess.graph.finalize()

        for i in range(constants.PARALLEL_SIZE):
            sess.run(training_threads[i].sync)
            training_threads[i].agent.reset()

        times = []

        if not constants.DEBUG and constants.RECORD_FEED_DICT:
            import h5py
            NUM_RECORD = 10000 * len(constants.USED_QUESTION_TYPES)
            time_str = py_util.get_time_str()
            if not os.path.exists('question_data_dump'):
                os.mkdir('question_data_dump')
            dataset = h5py.File('question_data_dump/maps_' + time_str + '.h5', 'w')

            dataset.create_dataset('question_data/existence_answer_placeholder', (NUM_RECORD, 1), dtype=np.int32)
            dataset.create_dataset('question_data/counting_answer_placeholder', (NUM_RECORD, 1), dtype=np.int32)

            dataset.create_dataset('question_data/question_type_placeholder', (NUM_RECORD, 1), dtype=np.int32)
            dataset.create_dataset('question_data/question_object_placeholder', (NUM_RECORD, 1), dtype=np.int32)
            dataset.create_dataset('question_data/question_container_placeholder', (NUM_RECORD, 21), dtype=np.int32)

            dataset.create_dataset('question_data/pose_placeholder', (NUM_RECORD, 3), dtype=np.int32)
            dataset.create_dataset('question_data/image_placeholder', (NUM_RECORD, 300, 300, 3), dtype=np.uint8)
            dataset.create_dataset('question_data/map_mask_placeholder',
                    (NUM_RECORD, constants.SPATIAL_MAP_HEIGHT, constants.SPATIAL_MAP_WIDTH, 27), dtype=np.uint8)
            dataset.create_dataset('question_data/meta_action_placeholder', (NUM_RECORD, 7), dtype=np.int32)
            dataset.create_dataset('question_data/possible_move_placeholder', (NUM_RECORD, 31), dtype=np.int32)

            dataset.create_dataset('question_data/answer_weight', (NUM_RECORD, 1), dtype=np.int32)
            dataset.create_dataset('question_data/taken_action', (NUM_RECORD, 32), dtype=np.int32)
            dataset.create_dataset('question_data/new_episode', (NUM_RECORD, 1), dtype=np.int32)

        time_lock = threading.Lock()

        data_ind = 0

        def train_function(parallel_index):
            global global_t
            global dataset
            global data_ind
            print('----------------------------------------thread', parallel_index, 'global_t', global_t)
            training_thread = training_threads[parallel_index]
            last_global_t = global_t
            last_global_t_image = global_t

            while global_t < constants.MAX_TIME_STEP:
                diff_global_t, ep_length, ep_reward, num_unrolls, feed_dict = training_thread.process(global_t, summary_writer,
                        summary_ops, summary_placeholders)
                time_lock.acquire()

                if not constants.DEBUG and constants.RECORD_FEED_DICT:
                    print('    NEW ENTRY: %d %s' % (data_ind, training_thread.agent.game_state.scene_name))
                    dataset['question_data/new_episode'][data_ind] = 1
                    for k, v in feed_dict.items():
                        key = 'question_data/' + k.name.split('/')[-1].split(':')[0]
                        if any([s in key for s in {
                            'gru_placeholder', 'num_unrolls',
                            'reward_placeholder', 'td_placeholder',
                            'learning_rate', 'phi_hat_prev_placeholder',
                            'next_map_mask_placeholder', 'next_pose_placeholder',
                            'episode_length_placeholder', 'question_count_placeholder',
                            'supervised_action_labels', 'supervised_action_weights_sigmoid',
                            'supervised_action_weights', 'question_direction_placeholder',
                        }]):
                            continue
                        v = np.ascontiguousarray(v)
                        if v.shape[0] == 1:
                            v = v[0, ...]
                        if len(v.shape) == 1 and num_unrolls > 1:
                            v = v[:, np.newaxis]
                        if 'map_mask_placeholder' in key:
                            v[:, :, :, :2] *= constants.STEPS_AHEAD
                            v[:, :, :, :2] += 2
                        data_loc = dataset[key][data_ind:data_ind + num_unrolls, ...]
                        data_len = data_loc.shape[0]
                        dataset[key][data_ind:data_ind + num_unrolls, ...] = v[:data_len, ...]
                    dataset.flush()

                    data_ind += num_unrolls
                    if data_ind >= NUM_RECORD:
                        # Everything is done
                        dataset.close()
                        raise Exception('Fake exception to exit the process. Don\'t worry, everything is fine.')
                if ep_length > 0:
                    times.append((ep_length, ep_reward))
                    print('Num episodes', len(times), 'Episode means', np.mean(times, axis=0))
                time_lock.release()
                global_t += diff_global_t
                # periodically save checkpoints to disk
                if not (constants.DEBUG or constants.RECORD_FEED_DICT) and parallel_index == 0:
                    if global_t - last_global_t_image > 10000:
                        print('Ran conv image summary')
                        summary_im_str = sess.run(conv_image_summary)
                        summary_writer.add_summary(summary_im_str, global_t)
                        last_global_t_image = global_t
                    if global_t - last_global_t > 10000:
                        print('Save checkpoint at timestamp %d' % global_t)
                        tf_util.save(saver, constants.CHECKPOINT_DIR, global_t)
                        last_global_t = global_t

        def test_function():
            global global_t
            last_global_t_test = 0
            while global_t < constants.MAX_TIME_STEP:
                time.sleep(1)
                if global_t - last_global_t_test > 10000:
                    # RUN TEST
                    sess.run(testing_thread.sync)
                    from game_state import QuestionGameState
                    if testing_thread.agent.game_state is None:
                        testing_thread.agent.game_state = QuestionGameState(sess=sess)
                    for q_type in constants.USED_QUESTION_TYPES:
                        answers_correct = 0.0
                        ep_lengths = 0.0
                        ep_rewards = 0.0
                        invalid_percents = 0.0
                        rows = list(range(len(testing_thread.agent.game_state.test_datasets[q_type])))
                        random.shuffle(rows)
                        rows = rows[:16]
                        print('()()()()()()()rows', rows)
                        for rr,row in enumerate(rows):
                            answer_correct, answer, gt_answer, ep_length, ep_reward, invalid_percent, scene_num, seed, required_interaction = testing_thread.process((row, q_type))
                            answers_correct += int(answer_correct)
                            ep_lengths += ep_length
                            ep_rewards += ep_reward
                            invalid_percents += invalid_percent
                            print('############################### TEST ITERATION ##################################')
                            print('ep ', (rr + 1))
                            print('average correct', answers_correct / (rr + 1))
                            print('#################################################################################')
                        answers_correct /= len(rows)
                        ep_lengths /= len(rows)
                        ep_rewards /= len(rows)
                        invalid_percents /= len(rows)

                        # Write the summary
                        test_summary_str = sess.run(test_summary_ops[q_type], feed_dict={
                            test_episode_reward_input : ep_rewards,
                            test_episode_length_input : ep_lengths,
                            test_exist_answer_correct_input : answers_correct,
                            test_count_answer_correct_input : answers_correct,
                            test_contains_answer_correct_input : answers_correct,
                            test_percent_invalid_actions_input : invalid_percents,
                            })
                        summary_writer.add_summary(test_summary_str, global_t)
                        summary_writer.flush()
                        last_global_t_test = global_t
                    testing_thread.agent.game_state.env.stop()
                    testing_thread.agent.game_state = None



        train_threads = []
        for i in range(constants.PARALLEL_SIZE):
            train_threads.append(threading.Thread(target=train_function, args=(i,)))
            train_threads[i].daemon = True

        for t in train_threads:
            t.start()

        if constants.RUN_TEST:
            test_thread = threading.Thread(target=test_function)
            test_thread.daemon = True
            test_thread.start()

        for t in train_threads:
            t.join()

        if constants.RUN_TEST:
            test_thread.join()

        if not constants.DEBUG:
            if not os.path.exists(constants.CHECKPOINT_DIR):
                os.makedirs(constants.CHECKPOINT_DIR)
            saver.save(sess, constants.CHECKPOINT_DIR + '/' + 'checkpoint', global_step = global_t)
            summary_writer.close()
            print('Saved.')

    except KeyboardInterrupt:
        print('Press Ctrl+C to stop')
    except:
        import traceback
        traceback.print_exc()
    finally:
        if not constants.DEBUG:
            print('Now saving data. Please wait')
            tf_util.save(saver, sess, constants.CHECKPOINT_DIR, global_t)
            summary_writer.close()
            print('Saved.')

if __name__ == '__main__':
    run()
