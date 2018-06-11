# Train the initial answerer using supervised loss.

import numpy as np
import pdb
import random
import tensorflow as tf
import time
import glob
import os

from networks.qa_planner_network import QAPlannerNetwork
from utils import tf_util

import constants


def run():
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(constants.GPU_ID)
        with tf.variable_scope('global_network'):
            network = QAPlannerNetwork(constants.RL_GRU_SIZE, int(constants.BATCH_SIZE), 1)
            network.create_net()
            training_step = network.training(network.rl_total_loss)

        conv_var_list = [v for v in tf.trainable_variables() if 'conv' in v.name and 'weight' in v.name and
                        (v.get_shape().as_list()[0] != 1 or v.get_shape().as_list()[1] != 1)]
        for var in conv_var_list:
            tf_util.conv_variable_summaries(var, scope=var.name.replace('/', '_')[:-2])
        summary_with_images = tf.summary.merge_all()

        with tf.variable_scope('supervised_loss'):
            loss_ph = tf.placeholder(tf.float32)
            accs_ph = [tf.placeholder(tf.float32) for _ in range(4)]
            loss_summary_op = tf.summary.merge([
                tf.summary.scalar('supervised_loss', loss_ph),
                tf.summary.scalar('acc_1_exist', accs_ph[0]),
                tf.summary.scalar('acc_2_count', accs_ph[1]),
                tf.summary.scalar('acc_3_contains', accs_ph[2]),
                ])

        # prepare session
        sess = tf_util.Session()
        sess.run(tf.global_variables_initializer())

        if not (constants.DEBUG or constants.DRAWING):
            from utils import py_util
            time_str = py_util.get_time_str()
            summary_writer = tf.summary.FileWriter(os.path.join(constants.LOG_FILE, time_str), sess.graph)
        else:
            summary_writer = None

        # init or load checkpoint with saver
        saver = tf.train.Saver(max_to_keep=3)
        start_it = tf_util.restore_from_dir(sess, constants.CHECKPOINT_DIR)

        sess.graph.finalize()

        import h5py
        h5_file = sorted(glob.glob('question_data_dump/*.h5'), key=os.path.getmtime)[-1]
        dataset = h5py.File(h5_file)
        num_entries = np.sum(np.sum(dataset['question_data/pose_placeholder'][...], axis=1) > 0)
        print('num_entries', num_entries)
        start_inds = dataset['question_data/new_episode'][:num_entries]
        start_inds = np.where(start_inds[1:] != 1)[0]

        curr_it = 0
        data_time_total = 0
        solver_time_total = 0
        total_time_total = 0

        for iteration in range(start_it, constants.MAX_TIME_STEP):
            if iteration == start_it or iteration % 10 == 1:
                current_time_start = time.time()
            t_start = time.time()

            rand_inds = np.sort(np.random.choice(start_inds, int(constants.BATCH_SIZE), replace=False))
            rand_inds = rand_inds.tolist()

            existence_answer_placeholder = dataset['question_data/existence_answer_placeholder'][rand_inds]
            counting_answer_placeholder = dataset['question_data/counting_answer_placeholder'][rand_inds]

            question_type_placeholder = dataset['question_data/question_type_placeholder'][rand_inds]
            question_object_placeholder = dataset['question_data/question_object_placeholder'][rand_inds]
            question_container_placeholder = dataset['question_data/question_container_placeholder'][rand_inds]

            pose_placeholder = dataset['question_data/pose_placeholder'][rand_inds]
            image_placeholder = dataset['question_data/image_placeholder'][rand_inds]
            map_mask_placeholder = np.ascontiguousarray(dataset['question_data/map_mask_placeholder'][rand_inds])

            meta_action_placeholder = dataset['question_data/meta_action_placeholder'][rand_inds]
            possible_move_placeholder = dataset['question_data/possible_move_placeholder'][rand_inds]

            taken_action = dataset['question_data/taken_action'][rand_inds]
            answer_weight = np.ones((constants.BATCH_SIZE, 1))

            map_mask_placeholder = np.ascontiguousarray(map_mask_placeholder, dtype=np.float32)
            map_mask_placeholder[:, :, :, :2] -= 2
            map_mask_placeholder[:, :, :, :2] /= constants.STEPS_AHEAD

            map_mask_starts = map_mask_placeholder.copy()

            for bb in range(0, constants.BATCH_SIZE):
                object_ind = int(question_object_placeholder[bb])
                question_type_ind = int(question_type_placeholder[bb])
                if question_type_ind in {2, 3}:
                    container_ind = np.argmax(question_container_placeholder[bb])

                max_map_inds = np.argmax(map_mask_placeholder[bb, ...], axis=2)
                map_range = np.where(max_map_inds > 0)
                map_range_x = (np.min(map_range[1]), np.max(map_range[1]))
                map_range_y = (np.min(map_range[0]), np.max(map_range[0]))
                for jj in range(random.randint(0, 100)):
                    tmp_patch_start = (random.randint(map_range_x[0], map_range_x[1]),
                            random.randint(map_range_y[0], map_range_y[1]))
                    tmp_patch_end = (random.randint(map_range_x[0], map_range_x[1]),
                            random.randint(map_range_y[0], map_range_y[1]))
                    patch_start = (min(tmp_patch_start[0], tmp_patch_end[0]),
                            min(tmp_patch_start[1], tmp_patch_end[1]))
                    patch_end = (max(tmp_patch_start[0], tmp_patch_end[0]),
                            max(tmp_patch_start[1], tmp_patch_end[1]))

                    patch = map_mask_placeholder[bb, patch_start[1]:patch_end[1], patch_start[0]:patch_end[0], :]
                    if question_type_ind in {2, 3}:
                        obj_mem = patch[:, :, 6 + container_ind] + patch[:, :, 6 + object_ind]
                    else:
                        obj_mem = patch[:, :, 6 + object_ind].copy()
                    obj_mem += patch[:, :, 2]  # make sure seen locations stay marked.
                    if patch.size > 0 and np.max(obj_mem) == 0:
                        map_mask_placeholder[bb, patch_start[1]:patch_end[1], patch_start[0]:patch_end[0], 6:] = 0
            feed_dict = {
                    network.existence_answer_placeholder: np.ascontiguousarray(existence_answer_placeholder),
                    network.counting_answer_placeholder: np.ascontiguousarray(counting_answer_placeholder),

                    network.question_type_placeholder: np.ascontiguousarray(question_type_placeholder),
                    network.question_object_placeholder: np.ascontiguousarray(question_object_placeholder),
                    network.question_container_placeholder: np.ascontiguousarray(question_container_placeholder),
                    network.question_direction_placeholder: np.zeros((constants.BATCH_SIZE, 4), dtype=np.float32),

                    network.pose_placeholder: np.ascontiguousarray(pose_placeholder),
                    network.image_placeholder: np.ascontiguousarray(image_placeholder),
                    network.map_mask_placeholder: map_mask_placeholder,
                    network.meta_action_placeholder: np.ascontiguousarray(meta_action_placeholder),
                    network.possible_move_placeholder: np.ascontiguousarray(possible_move_placeholder),

                    network.taken_action: np.ascontiguousarray(taken_action),
                    network.answer_weight: np.ascontiguousarray(answer_weight),
                    network.episode_length_placeholder: np.ones((constants.BATCH_SIZE)),
                    network.question_count_placeholder: np.zeros((constants.BATCH_SIZE)),
                    }
            new_feed_dict = {}
            for key,value in feed_dict.items():
                if len(value.squeeze().shape) > 1:
                    new_feed_dict[key] = np.reshape(value, [int(constants.BATCH_SIZE), 1] + list(value.squeeze().shape[1:]))
                else:
                    new_feed_dict[key] = np.reshape(value, [int(constants.BATCH_SIZE), 1])
            feed_dict = new_feed_dict
            feed_dict[network.taken_action] = np.reshape(feed_dict[network.taken_action], (constants.BATCH_SIZE, -1))
            feed_dict[network.gru_placeholder] = np.zeros((int(constants.BATCH_SIZE), constants.RL_GRU_SIZE))

            data_t_end = time.time()
            if constants.DEBUG or constants.DRAWING:
                outputs = sess.run(
                        [training_step, network.rl_total_loss, network.existence_answer, network.counting_answer,
                            network.possible_moves, network.memory_crops_rot, network.taken_action],
                        feed_dict=feed_dict)
            else:
                if iteration == start_it + 10:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    outputs = sess.run([training_step, network.rl_total_loss, summary_with_images],
                            feed_dict=feed_dict,
                            options=run_options,
                            run_metadata=run_metadata)
                    loss_summary = outputs[2]
                    summary_writer.add_run_metadata(run_metadata, 'step_%07d' % iteration)
                    summary_writer.add_summary(loss_summary, iteration)
                    summary_writer.flush()
                elif iteration % 10 == 0:
                    if iteration % 100 == 0:
                        outputs = sess.run(
                                [training_step, network.rl_total_loss, summary_with_images],
                                feed_dict=feed_dict)
                        loss_summary = outputs[2]
                    elif iteration % 10 == 0:
                        outputs = sess.run(
                                [training_step, network.rl_total_loss,
                                 network.existence_answer,
                                 network.counting_answer,
                                 ],
                                feed_dict=feed_dict)
                        outputs[2] = outputs[2].reshape(-1, 1)
                        acc_q0 = np.sum((existence_answer_placeholder == (outputs[2] > 0.5)) *
                            (question_type_placeholder == 0)) / np.maximum(1, np.sum(question_type_placeholder == 0))
                        acc_q1 = np.sum((counting_answer_placeholder == np.argmax(outputs[3], axis=1)[..., np.newaxis]) *
                            (question_type_placeholder == 1)) / np.maximum(1, np.sum(question_type_placeholder == 1))
                        acc_q2 = np.sum((existence_answer_placeholder == (outputs[2] > 0.5)) *
                            (question_type_placeholder == 2)) / np.maximum(1, np.sum(question_type_placeholder == 2))
                        acc_q3 = np.sum((existence_answer_placeholder == (outputs[2] > 0.5)) *
                            (question_type_placeholder == 3)) / np.maximum(1, np.sum(question_type_placeholder == 3))

                        curr_loss = outputs[1]
                        outputs = sess.run([loss_summary_op],
                                feed_dict={
                                    accs_ph[0]: acc_q0,
                                    accs_ph[1]: acc_q1,
                                    accs_ph[2]: acc_q2,
                                    accs_ph[3]: acc_q3,
                                    loss_ph: curr_loss,
                                    })

                        loss_summary = outputs[0]
                        outputs.append(curr_loss)
                    summary_writer.add_summary(loss_summary, iteration)
                    summary_writer.flush()
                else:
                    outputs = sess.run([training_step, network.rl_total_loss],
                            feed_dict=feed_dict)

            loss = outputs[1]
            solver_t_end = time.time()

            if constants.DEBUG or constants.DRAWING:
                # Look at outputs
                guess_bool = outputs[2].flatten()
                guess_count = outputs[3]
                possible_moves_pred = outputs[4]
                memory_crop = outputs[5]
                print('loss', loss)
                for bb in range(constants.BATCH_SIZE):
                    if constants.DRAWING:
                        import cv2
                        import scipy.misc
                        from utils import drawing
                        object_ind = int(question_object_placeholder[bb])
                        question_type_ind = question_type_placeholder[bb]
                        if question_type_ind == 1:
                            answer = counting_answer_placeholder[bb]
                            guess = guess_count[bb]
                        else:
                            answer = existence_answer_placeholder[bb]
                            guess = np.concatenate(([1 - guess_bool[bb]], [guess_bool[bb]]))
                        if question_type_ind[0] in {2, 3}:
                            container_ind = np.argmax(question_container_placeholder[bb])
                            obj_mem = np.flipud(map_mask_placeholder[bb, :, :, 6 + container_ind]).copy()
                            obj_mem += 2 * np.flipud(map_mask_placeholder[bb, :, :, 6 + object_ind])
                        else:
                            obj_mem = np.flipud(map_mask_placeholder[bb, :, :, 6 + object_ind])

                        possible = possible_move_placeholder[bb,...].flatten()
                        possible = np.concatenate((possible, np.zeros(4)))
                        possible = possible.reshape(constants.STEPS_AHEAD + 2, constants.STEPS_AHEAD)

                        possible_pred = possible_moves_pred[bb,...].flatten()
                        possible_pred = np.concatenate((possible_pred, np.zeros(4)))
                        possible_pred = possible_pred.reshape(constants.STEPS_AHEAD + 2, constants.STEPS_AHEAD)

                        mem2 = np.flipud(np.argmax(memory_crop[bb,...], axis=2))
                        mem2[0, 0] = memory_crop.shape[-1] - 2

                        # Answer histogram
                        ans = guess
                        if len(ans) == 1:
                            ans = [ans[0], 1 - ans[0]]
                        ans_size = max(len(ans), 100)
                        ans_hist = np.zeros((ans_size, ans_size))
                        for ii,ans_i in enumerate(ans):
                            ans_hist[:max(int(np.round(ans_i * ans_size)), 1),
                                    int(ii * ans_size / len(ans)):int((ii+1) * ans_size / len(ans))] = (ii + 1)
                        ans_hist = np.flipud(ans_hist)

                        image_list = [
                                image_placeholder[bb,...],
                                ans_hist,
                                np.flipud(possible),
                                np.flipud(possible_pred),
                                mem2,
                                np.flipud(np.argmax(map_mask_starts[bb, :, :, 2:], axis=2)),
                                obj_mem,
                                ]
                        if question_type_ind == 0:
                            question_str = 'Ex Q: %s A: %s' % (constants.OBJECTS[object_ind], bool(answer))
                        elif question_type_ind == 1:
                            question_str = '# Q: %s A: %d' % (constants.OBJECTS[object_ind], answer)
                        elif question_type_ind == 2:
                            question_str = 'Q: %s in %s A: %s' % (
                                    constants.OBJECTS[object_ind],
                                    constants.OBJECTS[container_ind],
                                    bool(answer))
                        image = drawing.subplot(image_list, 4, 2, constants.SCREEN_WIDTH, constants.SCREEN_HEIGHT,
                                titles=[question_str, 'A: %s' % str(np.argmax(guess))], border=2)
                        cv2.imshow('image', image[:, :, ::-1])
                        cv2.waitKey(0)
                    else:
                        pdb.set_trace()

            if not (constants.DEBUG or constants.DRAWING) and (iteration % 1000 == 0 or iteration == constants.MAX_TIME_STEP - 1):
                saver_t_start = time.time()
                tf_util.save(saver, sess, constants.CHECKPOINT_DIR, iteration)
                saver_t_end = time.time()
                print('Saver:     %.3f' % (saver_t_end - saver_t_start))

            curr_it += 1

            data_time_total += data_t_end - t_start
            solver_time_total += solver_t_end - data_t_end
            total_time_total += time.time() - t_start

            if iteration == start_it or (iteration) % 10 == 0:
                print('Iteration: %d' % (iteration))
                print('Loss:      %.3f' % loss)
                print('Data:      %.3f' % (data_time_total / curr_it))
                print('Solver:    %.3f' % (solver_time_total / curr_it))
                print('Total:     %.3f' % (total_time_total / curr_it))
                print('Current:   %.3f\n' % ((time.time() - current_time_start) / min(10, curr_it)))

    except:
        import traceback
        traceback.print_exc()
    finally:
        # Save final model
        if not (constants.DEBUG or constants.DRAWING):
            tf_util.save(saver, sess, constants.CHECKPOINT_DIR, iteration)


if __name__ == '__main__':
    run()

