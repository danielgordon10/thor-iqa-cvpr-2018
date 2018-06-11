# Trains the navigation module using random start and end points.

import numpy as np
import pdb
import os
import random
import threading
import tensorflow as tf
import time

from networks.free_space_network import FreeSpaceNetwork
from supervised.sequence_generator import SequenceGenerator
from utils import tf_util

import constants

data_buffer = []
data_counts = np.full(constants.REPLAY_BUFFER_SIZE, 9999)
os.environ["CUDA_VISIBLE_DEVICES"] = str(constants.GPU_ID)


def run():
    try:
        with tf.variable_scope('nav_global_network'):
            network = FreeSpaceNetwork(constants.GRU_SIZE, constants.BATCH_SIZE, constants.NUM_UNROLLS)
            network.create_net()
            training_step = network.training_op

        with tf.variable_scope('loss'):
            loss_summary_op = tf.summary.merge([
                tf.summary.scalar('loss', network.loss),
                ])
        summary_full = tf.summary.merge_all()
        conv_var_list = [v for v in tf.trainable_variables() if 'conv' in v.name and 'weight' in v.name and
                        (v.get_shape().as_list()[0] != 1 or v.get_shape().as_list()[1] != 1)]
        for var in conv_var_list:
            tf_util.conv_variable_summaries(var, scope=var.name.replace('/', '_')[:-2])
        summary_with_images = tf.summary.merge_all()

        # prepare session
        sess = tf_util.Session()

        seq_inds = np.zeros((constants.BATCH_SIZE, 2), dtype=np.int32)

        sequence_generators = []
        for thread_index in range(constants.PARALLEL_SIZE):
            gpus = str(constants.GPU_ID).split(',')
            sequence_generator = SequenceGenerator(sess)
            sequence_generators.append(sequence_generator)

        sess.run(tf.global_variables_initializer())

        if not (constants.DEBUG or constants.DRAWING):
            from utils import py_util
            time_str = py_util.get_time_str()
            summary_writer = tf.summary.FileWriter(os.path.join(constants.LOG_FILE, time_str), sess.graph)
        else:
            summary_writer = None

        saver = tf.train.Saver(max_to_keep=3)

        # init or load checkpoint
        start_it = tf_util.restore_from_dir(sess, constants.CHECKPOINT_DIR)

        sess.graph.finalize()

        data_lock = threading.Lock()
        def load_new_data(thread_index):
            global data_buffer
            global data_counts

            sequence_generator = sequence_generators[thread_index]
            counter = 0
            while True:
                while not (len(data_buffer) < constants.REPLAY_BUFFER_SIZE or np.max(data_counts) > 0):
                    time.sleep(1)
                counter += 1
                if constants.DEBUG:
                    print('\nThread %d' % thread_index)
                new_data, bounds, goal_pose = sequence_generator.generate_episode()
                new_data = {key : ([new_data[ii][key] for ii in range(len(new_data))]) for key in new_data[0]}
                new_data['goal_pose'] = goal_pose
                new_data['memory'] = np.zeros(
                    (constants.SPATIAL_MAP_HEIGHT, constants.SPATIAL_MAP_WIDTH, constants.MEMORY_SIZE))
                new_data['gru_state'] = np.zeros(constants.GRU_SIZE)
                if constants.DRAWING:
                    new_data['debug_images'] = sequence_generator.debug_images
                data_lock.acquire()
                if len(data_buffer) < constants.REPLAY_BUFFER_SIZE:
                    data_counts[len(data_buffer)] = 0
                    data_buffer.append(new_data)
                    counts = data_counts[:len(data_buffer)]
                    if counter % 10 == 0:
                        print('Buffer size %d  Num used %d  Max used amount %d' % (len(data_buffer), len(counts[counts > 0]), np.max(counts)))
                else:
                    max_count_ind = np.argmax(data_counts)
                    data_buffer[max_count_ind] = new_data
                    data_counts[max_count_ind] = 0
                    if counter % 10 == 0:
                        print('Num used %d  Max used amount %d' % (len(data_counts[data_counts > 0]), np.max(data_counts)))
                data_lock.release()

        threads = []
        for i in range(constants.PARALLEL_SIZE):
            load_data_thread = threading.Thread(target=load_new_data, args=(i,))
            load_data_thread.daemon = True
            load_data_thread.start()
            threads.append(load_data_thread)
            time.sleep(1)

        sequences = [None] * constants.BATCH_SIZE

        curr_it = 0
        dataTimeTotal = 0.00001
        solverTimeTotal = 0.00001
        summaryTimeTotal = 0.00001
        totalTimeTotal = 0.00001

        chosen_inds = set()
        loc_to_chosen_ind = {}
        for iteration in range(start_it, constants.MAX_TIME_STEP):
            if iteration == start_it or iteration % 10 == 1:
                currentTimeStart = time.time()
            tStart = time.time()
            batch_data = []
            batch_action = []
            batch_memory = []
            batch_gru_state = []
            batch_labels = []
            batch_pose = []
            batch_mask = []
            batch_goal_pose = []
            batch_pose_indicator = []
            batch_possible_label = []
            batch_debug_images = []
            for bb in range(constants.BATCH_SIZE):
                if seq_inds[bb, 0] == seq_inds[bb, 1]:
                    # Pick a new random sequence
                    pickable_inds = set(np.where(data_counts < 100)[0]) - chosen_inds
                    count_size = len(pickable_inds)
                    while count_size == 0:
                        pickable_inds = set(np.where(data_counts < 100)[0]) - chosen_inds
                        count_size = len(pickable_inds)
                        time.sleep(1)
                    random_ind = random.sample(pickable_inds, 1)[0]
                    data_lock.acquire()
                    sequences[bb] = data_buffer[random_ind]
                    goal_pose = sequences[bb]['goal_pose']
                    sequences[bb]['memory'] = np.zeros(
                            (constants.SPATIAL_MAP_HEIGHT, constants.SPATIAL_MAP_WIDTH, constants.MEMORY_SIZE))
                    sequences[bb]['gru_state'] = np.zeros(constants.GRU_SIZE)
                    data_counts[random_ind] += 1
                    if bb in loc_to_chosen_ind:
                        chosen_inds.remove(loc_to_chosen_ind[bb])
                    loc_to_chosen_ind[bb] = random_ind
                    chosen_inds.add(random_ind)
                    data_lock.release()
                    seq_inds[bb, 0] = 0
                    seq_inds[bb, 1] = len(sequences[bb]['color'])
                data_len = min(constants.NUM_UNROLLS, seq_inds[bb, 1] - seq_inds[bb, 0])
                ind0 = seq_inds[bb, 0]
                ind1 = seq_inds[bb, 0] + data_len
                data = sequences[bb]['color'][ind0:ind1]
                action = sequences[bb]['action'][ind0:ind1]
                labels = sequences[bb]['label'][ind0:ind1]
                memory = sequences[bb]['memory'].copy()
                gru_state = sequences[bb]['gru_state'].copy()
                pose = sequences[bb]['pose'][ind0:ind1]
                goal_pose = sequences[bb]['goal_pose']
                mask = sequences[bb]['weight'][ind0:ind1]
                pose_indicator = sequences[bb]['pose_indicator'][ind0:ind1]
                possible_label = sequences[bb]['possible_label'][ind0:ind1]
                if constants.DRAWING:
                    batch_debug_images.append(sequences[bb]['debug_images'][ind0:ind1])
                if data_len < (constants.NUM_UNROLLS):
                    seq_inds[bb, :] = 0
                    data.extend([np.zeros_like(data[0]) for _ in range(constants.NUM_UNROLLS - data_len)])
                    action.extend([np.zeros_like(action[0]) for _ in range(constants.NUM_UNROLLS - data_len)])
                    labels.extend([np.zeros_like(labels[0]) for _ in range(constants.NUM_UNROLLS - data_len)])
                    pose.extend([pose[-1] for _ in range(constants.NUM_UNROLLS - data_len)])
                    mask.extend([np.zeros_like(mask[0]) for _ in range(constants.NUM_UNROLLS - data_len)])
                    pose_indicator.extend([np.zeros_like(pose_indicator[0]) for _ in range(constants.NUM_UNROLLS - data_len)])
                    possible_label.extend([np.zeros_like(possible_label[0]) for _ in range(constants.NUM_UNROLLS - data_len)])
                else:
                    seq_inds[bb, 0] += constants.NUM_UNROLLS
                batch_data.append(data)
                batch_action.append(action)
                batch_memory.append(memory)
                batch_gru_state.append(gru_state)
                batch_pose.append(pose)
                batch_goal_pose.append(goal_pose)
                batch_labels.append(labels)
                batch_mask.append(mask)
                batch_pose_indicator.append(pose_indicator)
                batch_possible_label.append(possible_label)

            feed_dict = {
                    network.image_placeholder: np.ascontiguousarray(batch_data),
                    network.action_placeholder: np.ascontiguousarray(batch_action),
                    network.gru_placeholder: np.ascontiguousarray(batch_gru_state),
                    network.pose_placeholder: np.ascontiguousarray(batch_pose),
                    network.goal_pose_placeholder: np.ascontiguousarray(batch_goal_pose),
                    network.labels_placeholder: np.ascontiguousarray(batch_labels)[...,np.newaxis],
                    network.mask_placeholder: np.ascontiguousarray(batch_mask),
                    network.pose_indicator_placeholder: np.ascontiguousarray(batch_pose_indicator),
                    network.possible_label_placeholder: np.ascontiguousarray(batch_possible_label),
                    network.memory_placeholders: np.ascontiguousarray(batch_memory),
                    }
            dataTEnd = time.time()
            summaryTime = 0
            if constants.DEBUG or constants.DRAWING:
                outputs = sess.run(
                    [training_step, network.loss, network.gru_state, network.patch_weights_sigm,
                     network.gru_outputs_full, network.is_possible_sigm,
                     network.pose_indicator_placeholder, network.terminal_patches,
                     network.gru_outputs],
                    feed_dict=feed_dict)
            else:
                if iteration == start_it + 10:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    outputs = sess.run([training_step, network.loss, network.gru_state,
                        summary_with_images, network.gru_outputs],
                            feed_dict=feed_dict,
                            options=run_options,
                            run_metadata=run_metadata)
                    loss_summary = outputs[3]
                    summary_writer.add_run_metadata(run_metadata, 'step_%07d' % iteration)
                    summary_writer.add_summary(loss_summary, iteration)
                    summary_writer.flush()
                elif iteration % 10 == 0:
                    if iteration % 100 == 0:
                        outputs = sess.run(
                                [training_step, network.loss, network.gru_state, summary_with_images, network.gru_outputs],
                                feed_dict=feed_dict)
                    elif iteration % 10 == 0:
                        outputs = sess.run(
                                [training_step, network.loss, network.gru_state, loss_summary_op, network.gru_outputs],
                                feed_dict=feed_dict)
                    loss_summary = outputs[3]
                    summaryTStart = time.time()
                    summary_writer.add_summary(loss_summary, iteration)
                    summary_writer.flush()
                    summaryTime = time.time() - summaryTStart
                else:
                    outputs = sess.run([training_step, network.loss, network.gru_state, network.gru_outputs], feed_dict=feed_dict)

            gru_state_out = outputs[2]
            memory_out = outputs[-1]
            for mm in range(constants.BATCH_SIZE):
                sequences[mm]['memory'] = memory_out[mm,...]
                sequences[mm]['gru_state'] = gru_state_out[mm,...]

            loss = outputs[1]
            solverTEnd = time.time()

            if constants.DEBUG or constants.DRAWING:
                # Look at outputs
                patch_weights = outputs[3]
                is_possible = outputs[5]
                pose_indicator = outputs[6]
                terminal_patches = outputs[7]
                data_lock.acquire()
                for bb in range(constants.BATCH_SIZE):
                    for tt in range(constants.NUM_UNROLLS):
                        if batch_mask[bb][tt] == 0:
                                break
                        if constants.DRAWING:
                            import cv2
                            import scipy.misc
                            from utils import drawing
                            curr_image = batch_data[bb][tt]
                            label = np.flipud(batch_labels[bb][tt])
                            debug_images = batch_debug_images[bb][tt]
                            color_image = debug_images['color']
                            state_image = debug_images['state_image']
                            label_memory_image = debug_images['label_memory'][:,:,0]
                            label_memory_image_class = np.argmax(debug_images['label_memory'][:,:,1:], axis=2)
                            label_memory_image_class[0,0] = constants.NUM_CLASSES

                            label_patch = debug_images['label']

                            print('Possible pred %.3f' % is_possible[bb,tt])
                            print('Possible label %.3f' % batch_possible_label[bb][tt])
                            patch = np.flipud(patch_weights[bb,tt,...])
                            patch_occupancy = patch[:,:,0]
                            print('occ', patch_occupancy)
                            print('label', label)
                            terminal_patch = np.flipud(np.sum(terminal_patches[bb,tt,...], axis=2))
                            image_list = [
                                    debug_images['color'],
                                    state_image,
                                    debug_images['label_memory'][:,:,0],
                                    debug_images['memory_map'][:,:,0],
                                    label[:,:],
                                    patch_occupancy,
                                    np.flipud(pose_indicator[bb,tt]),
                                    terminal_patch,
                                    ]

                            image = drawing.subplot(image_list, 4, 2, constants.SCREEN_WIDTH, constants.SCREEN_HEIGHT)
                            cv2.imshow('image', image[:,:,::-1])
                            cv2.waitKey(0)
                        else:
                            pdb.set_trace()
                data_lock.release()

            if not (constants.DEBUG or constants.DRAWING) and (iteration % 500 == 0 or iteration == constants.MAX_TIME_STEP - 1):
                saverTStart = time.time()
                tf_util.save(saver, sess, constants.CHECKPOINT_DIR, iteration)
                saverTEnd = time.time()
                print('Saver:     %.3f' % (saverTEnd - saverTStart))

            curr_it += 1

            dataTimeTotal += dataTEnd - tStart
            summaryTimeTotal += summaryTime
            solverTimeTotal += solverTEnd - dataTEnd - summaryTime
            totalTimeTotal += time.time() - tStart

            if iteration == start_it or (iteration) % 10 == 0:
                print('Iteration: %d' % (iteration))
                print('Loss:      %.3f' % loss)
                print('Data:      %.3f' % (dataTimeTotal / curr_it))
                print('Solver:    %.3f' % (solverTimeTotal / curr_it))
                print('Summary:   %.3f' % (summaryTimeTotal / curr_it))
                print('Total:     %.3f' % (totalTimeTotal / curr_it))
                print('Current:   %.3f\n' % ((time.time() - currentTimeStart) / min(10, curr_it)))

    except:
        import traceback
        traceback.print_exc()
    finally:
        # Save final model
        if not (constants.DEBUG or constants.DRAWING):
            tf_util.save(saver, sess, constants.CHECKPOINT_DIR, iteration)

if __name__ == '__main__':
    run()

