# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

from utils import game_util
from qa_agents import graph_agent
from qa_agents import end_to_end_baseline_agent
import time

import constants

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

class A3CTestingThread(object):
    def __init__(self,
            thread_index,
            sess,
            free_space_network_scope,
            depth_network_scope):

        self.thread_index = thread_index
        self.agent = None


        time.sleep(0.1)
        with tf.device('/gpu:%s' % constants.GPU_ID):
            network_scope = 'worker_network_%02d' % self.thread_index
            with tf.variable_scope(network_scope):
                if constants.END_TO_END_BASELINE:
                    self.agent = end_to_end_baseline_agent.EndToEndBaselineGraphAgent(sess, num_unrolls=None, depth_scope=depth_network_scope)
                else:
                    self.agent = graph_agent.RLGraphAgent(sess, num_unrolls=None,
                                                          free_space_network_scope=free_space_network_scope, depth_scope=depth_network_scope)

        var_refs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=network_scope)
        global_var_refs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='global_network')

        self.sync = self.agent.network.sync_from(global_var_refs, var_refs)
        self.local_t = 0

    def process(self, test_ind):
        terminal = False
        episode_reward = 0

        print('Resetting ')
        self.prev_action = {'action' : 'Reset'}
        self.agent.reset(seed=test_ind[0], test_ind=test_ind)
        scene_num = self.agent.game_state.scene_num
        scene_seed = self.agent.game_state.scene_seed
        required_interaction = self.agent.game_state.requires_interaction

        while not terminal and self.agent.num_steps <= constants.MAX_EPISODE_LENGTH:
            self.local_t += 1
            if (self.agent.num_steps % 100) == 0:
                print("TIMESTEP", self.agent.num_steps)
            self.agent.inference()

            pi = self.agent.pi
            action = game_util.choose_action(pi)

            action_dict = self.agent.get_action(action)

            if constants.DRAWING:
                self.draw_frame(self.prev_action)
                self.prev_action = action_dict

            # process game
            self.agent.step(action_dict)
            reward, terminal = self.agent.get_reward()

            if (self.agent.num_steps % 100) == 0:
                print("Pi =", pi)
                print("V =", self.agent.v)
                print('action', action_dict)
                print('reward', reward)
                print('terminal', terminal)

            episode_reward += reward

        if constants.DRAWING:
            self.agent.inference()
            self.draw_frame(self.prev_action)
            self.prev_action = {'action' : 'Reset'}
        print('-------------------- FINISHED --------------------')
        print('Episode number', test_ind)
        print("episode reward = %.3f" % episode_reward)
        print("episode length = %d"     % self.agent.num_steps)
        if self.agent.game_state.question_type_ind != 1:
            answer = (self.agent.answer > 0.5)
        else:
            answer = np.argmax(self.agent.answer)
        gt_answer = self.agent.game_state.answer
        correct = answer == gt_answer
        episode_length = self.agent.num_steps

        print("answer_correct", correct)
        invalid_percent = self.agent.num_invalid_actions * 1.0 / max(self.agent.num_steps, 1)
        print("percent_invalid_actions_input", self.agent.num_invalid_actions * 1.0 / max(self.agent.num_steps, 1))


        return correct, answer, gt_answer, episode_length, episode_reward, invalid_percent, scene_num, scene_seed, required_interaction

    def draw_frame(self, action):
        subplot = self.agent.draw_state(action=action)

