# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

from utils import game_util
from qa_agents import graph_agent
from qa_agents import end_to_end_baseline_agent
import time

import constants


class A3CTrainingThread(object):
    def __init__(self,
            thread_index,
            sess,
            learning_rate_input,
            grad_applier,
            max_global_time_step,
            free_space_network_scope,
            depth_network_scope):

        self.thread_index = thread_index
        self.learning_rate_input = learning_rate_input
        self.max_global_time_step = max_global_time_step
        self.agent = None

        self.local_t = 0
        self.count = 0

        self.episode_reward = 0
        self.episode_length = 0

        time.sleep(0.1)
        network_scope = 'worker_network_%02d' % self.thread_index
        with tf.variable_scope(network_scope):
            if constants.END_TO_END_BASELINE:
                self.agent = end_to_end_baseline_agent.EndToEndBaselineGraphAgent(sess, num_unrolls=None, depth_scope=depth_network_scope)
            else:
                self.agent = graph_agent.RLGraphAgent(sess, num_unrolls=None,
                                                      free_space_network_scope=free_space_network_scope, depth_scope=depth_network_scope)

        var_refs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=network_scope)
        global_var_refs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='global_network')

        self.gradients, _ = tf.clip_by_global_norm(tf.gradients(
            self.agent.network.rl_total_loss, var_refs,
            gate_gradients=False), 40)
        self.apply_gradients = grad_applier.apply_gradients(global_var_refs, self.gradients)
        self.sync = self.agent.network.sync_from(global_var_refs, var_refs)

        self.rl_loss_ph = tf.placeholder(tf.float32, name='rl_loss_ph')
        self.loss_summary = tf.summary.merge([
            tf.summary.scalar('rl_loss', self.rl_loss_ph),
            ])
        self.prev_action = {'action' : 'Reset'}

    def _record_score(self, writer, summary_op, placeholders, values, global_t):
        feed_dict = {}
        for k in placeholders:
            if k not in values:
                continue
            feed_dict[placeholders[k]] = values[k]
        summary_str = self.agent.sess.run(summary_op, feed_dict=feed_dict)
        print('writing to summary writer at time %d' % (global_t))
        writer.add_summary(summary_str, global_t)

    def process(self, global_t, summary_writer, summary_op, summary_placeholders):
        feed_dicts = []
        actions = []
        masks = []
        labels = []
        rewards = []
        values = []

        terminal = False

        # copy weights from shared to local
        if not (constants.DEBUG or constants.RECORD_FEED_DICT):
            self.agent.sess.run(self.sync)

        start_local_t = self.local_t

        start_gru_state = self.agent.gru_state.copy()
        ep_reward = 0
        ep_length = 0

        # t_max times loop
        for i in range(constants.LOCAL_T_MAX):
            if (self.thread_index == 0) and (self.local_t % 100) == 0:
                print("TIMESTEP", self.local_t)
            self.agent.inference()


            # Need this because reset can cause size of pi to change.
            pi = self.agent.pi
            if constants.RECORD_FEED_DICT:
                pi[-1] = 0
                pi /= np.sum(pi)
            action = game_util.choose_action(pi)
            values.append(self.agent.v.squeeze())

            action_one_hot = np.zeros(len(pi.squeeze()))
            action_one_hot[action] = 1
            self.agent.last_action_one_hot = action_one_hot
            actions.append(action_one_hot)
            if not constants.END_TO_END_BASELINE:
                masks.append(self.agent.map_mask_padded.copy())
            feed_dicts.append(self.agent.feed_dict)

            action_dict = self.agent.get_action(action)

            if constants.DRAWING:
                self.draw_frame(self.prev_action)
                self.prev_action = action_dict

            # process game
            self.agent.step(action_dict)

            reward, terminal = self.agent.get_reward()

            if (self.thread_index == 0) and (self.local_t % 100) == 0:
                print("Pi =", pi)
                print("V =", self.agent.v)
                print('action', action_dict)
                print('reward', reward)
                print('terminal', terminal)

            self.episode_reward += reward

            rewards.append(reward)

            self.local_t += 1

            if terminal:
                if constants.DRAWING:
                    self.agent.inference()
                    self.draw_frame(self.prev_action)
                    self.prev_action = {'action' : 'Reset'}
                self.local_t += 1
                print('-------------------- FINISHED --------------------')
                print("thread = %d global t = %d" % (self.thread_index, global_t))
                print("episode reward = %.3f" % self.episode_reward)
                print("episode length = %d"     % self.agent.num_steps)
                ep_reward = self.episode_reward
                if not constants.DEBUG:
                    summary_values = {
                        "episode_reward_input": self.episode_reward,
                        "episode_length_input": float(self.agent.num_steps),
                        "percent_invalid_actions_input": self.agent.num_invalid_actions * 1.0 / max(self.agent.num_steps, 1),
                    }
                    if self.agent.game_state.question_type_ind == 0:
                        summary_values["exist_answer_correct_input"] = int((self.agent.answer[0] > 0.5) == self.agent.game_state.answer)
                    elif self.agent.game_state.question_type_ind == 1:
                        summary_values["count_answer_correct_input"] = np.argmax(self.agent.answer) == self.agent.game_state.answer
                    if self.agent.game_state.question_type_ind == 2:
                        summary_values["contains_answer_correct_input"] = int((self.agent.answer[0] > 0.5) == self.agent.game_state.answer)
                    if self.agent.game_state.question_type_ind == 3:
                        summary_values["direction_answer_correct_input"] = int((self.agent.answer[0] > 0.5) == self.agent.game_state.answer)

                    self._record_score(summary_writer, summary_op[self.agent.game_state.question_type_ind],
                            summary_placeholders,
                            summary_values, global_t)

                self.episode_reward = 0
                ep_length = self.agent.num_steps
                print('Resetting')
                self.agent.reset()
                break

        R = 0.0
        if not terminal:
            R = self.agent.get_next_output()[1][0,...]

        rewards.reverse()
        values.reverse()

        batch_td = []
        batch_R = []

        # compute and accmulate gradients
        for(ri, Vi) in zip(rewards, values):
            R = ri + constants.GAMMA * R
            td = R - Vi
            batch_td.append(td)

            batch_R.append(R)

        feed_dict = {self.learning_rate_input: 0.001}
        batch_td.reverse()
        batch_R.reverse()
        feed_dict[self.agent.network.r] = batch_R
        feed_dict[self.agent.network.td] = batch_td
        # Get all inputs from inference.
        for key in feed_dicts[0].keys():
            if key == self.agent.network.num_unrolls or key == self.agent.network.gru_placeholder:
                continue
            feed_dict[key] = np.concatenate([fd[key] for fd in feed_dicts], axis=1)
        # Reset memory state to initial memory state.
        feed_dict[self.agent.network.gru_placeholder] = start_gru_state
        feed_dict[self.agent.network.num_unrolls] = len(actions)
        feed_dict[self.agent.network.taken_action] = actions

        if terminal:
            if self.agent.network.answer_weight is not None:
                feed_dict[self.agent.network.answer_weight] = np.ones_like(feed_dict[self.agent.network.answer_weight])

        if not constants.END_TO_END_BASELINE:
            feed_dict[self.agent.network.map_mask_placeholder] = np.array(masks)[np.newaxis,...]

        if not constants.RECORD_FEED_DICT:
            if not constants.DEBUG:
                if self.count % 10 == 0:
                    print('\tthread', self.thread_index, 'still alive')
                if (self.thread_index == 0) and self.count % 1 == 0:
                    if self.thread_index == 0 and self.count % 1e6 == 0:
                        print('Running full graph summary.')
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        _, rl_loss = self.agent.sess.run([
                            self.apply_gradients,
                            self.agent.network.rl_total_loss,
                            ],
                                feed_dict=feed_dict,
                                options=run_options,
                                run_metadata=run_metadata)
                        summary_writer.add_run_metadata(run_metadata, 'step_%07d' % global_t)
                    else:
                        _, rl_loss = self.agent.sess.run([self.apply_gradients,
                                self.agent.network.rl_total_loss,
                                ], feed_dict=feed_dict)
                    loss_summary_str = self.agent.sess.run(self.loss_summary, feed_dict={self.rl_loss_ph : rl_loss})
                    summary_writer.add_summary(loss_summary_str, global_t)
                    summary_writer.flush()

                else:
                    _, rl_loss = self.agent.sess.run([self.apply_gradients, self.agent.network.rl_total_loss], feed_dict=feed_dict)
            else:
                print('running full forward pass')
                rl_loss, taken_action, _ = self.agent.sess.run([
                    self.agent.network.rl_total_loss,
                    self.agent.network.taken_action,
                    self.apply_gradients,
                    ], feed_dict=feed_dict)
            self.count += 1

            if terminal:
                print('-------------------------------------------------')

        diff_local_t = self.local_t - start_local_t
        return diff_local_t, ep_length, ep_reward, len(actions), feed_dict

    def draw_frame(self, action):
        subplot = self.agent.draw_state(action=action)
        return subplot
