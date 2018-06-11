import pdb
import cv2
import numpy as np
import scipy.ndimage
import os
import tensorflow as tf
import time

from networks.free_space_network import FreeSpaceNetwork
from game_state import GameState
from utils import game_util
from graph import graph_obj
from networks.qa_planner_network import QAPlannerNetwork
from game_state import QuestionGameState
from qa_agents.qa_agent import QAAgent

import constants


class GraphAgent(object):
    def __init__(self, sess, reuse=True, num_unrolls=1, game_state=None, net_scope=None):
        if net_scope is None:
            with tf.name_scope('agent'):
                with tf.variable_scope('nav_global_network', reuse=reuse):
                    self.network = FreeSpaceNetwork(constants.GRU_SIZE, 1, num_unrolls)
                    self.network.create_net()
        else:
            with tf.variable_scope(net_scope, reuse=True):
                self.network = FreeSpaceNetwork(constants.GRU_SIZE, 1, 1)
                self.network.create_net(add_loss=False)

        if game_state is None:
            self.game_state = GameState(sess=sess)
        else:
            self.game_state = game_state
        self.action_util = self.game_state.action_util
        self.gt_graph = None
        self.sess = sess
        self.num_steps = 0
        self.global_step_id = 0
        self.num_unrolls = num_unrolls
        self.pose_indicator = np.zeros((constants.TERMINAL_CHECK_PADDING * 2 + 1,
                                        constants.TERMINAL_CHECK_PADDING * 2 + 1))
        self.times = np.zeros(2)

    def goto(self, action, step_num):
        # Look down
        start_angle = self.game_state.pose[3]
        if start_angle != 60:
            look_action = {'action': 'TeleportFull',
                           'x': self.game_state.pose[0] * constants.AGENT_STEP_SIZE,
                           'y': self.game_state.agent_height,
                           'z': self.game_state.pose[1] * constants.AGENT_STEP_SIZE,
                           'rotateOnTeleport': True,
                           'rotation': self.game_state.pose[2] * 90,
                           'horizon': 60,
                          }
            super(QuestionGameState, self.game_state).step(look_action)

        self.game_state.end_point = (action['x'], action['z'], action['rotation'] / 90)
        self.goal_pose = np.array([self.game_state.end_point[0] - self.game_state.graph.xMin,
                self.game_state.end_point[1] - self.game_state.graph.yMin],
                dtype=np.int32)[:2]
        self.pose = self.game_state.pose
        self.inference()
        plan, path = self.get_plan()
        steps = 0
        invalid_actions = 0

        self.reset(self.game_state.scene_name)

        self.game_state.board = None
        while steps < 20 and len(plan) > 0 and self.is_possible >= constants.POSSIBLE_THRESH:
            t_start = time.time()
            action = plan[0]
            self.step(action)
            invalid_actions += 1 - int(self.game_state.event.metadata['lastActionSuccess'])

            plan, path = self.get_plan()
            steps += 1
            if constants.DRAWING:
                image = self.draw_state()
                if not os.path.exists('visualizations/images'):
                    os.makedirs('visualizations/images')
                cv2.imwrite('visualizations/images/state_%05d.jpg' %
                        (step_num + steps), image[:, :, ::-1])
            self.times[0] += time.time() - t_start

        print('step time %.3f' % (self.times[0] / max(steps, 1)))
        self.times[0] = 0

        # Look back
        if start_angle != 60:
            look_action = {'action': 'TeleportFull',
                           'x': self.game_state.pose[0] * constants.AGENT_STEP_SIZE,
                           'y': self.game_state.agent_height,
                           'z': self.game_state.pose[1] * constants.AGENT_STEP_SIZE,
                           'rotateOnTeleport': True,
                           'rotation': self.game_state.pose[2] * 90,
                           'horizon': start_angle,
                           }
            super(QuestionGameState, self.game_state).step(look_action)
        return steps, invalid_actions

    def inference(self):
        image = self.game_state.s_t[np.newaxis, np.newaxis, ...]

        self.pose_indicator = np.zeros((constants.TERMINAL_CHECK_PADDING * 2 + 1, constants.TERMINAL_CHECK_PADDING * 2 + 1))
        if (abs(self.pose[0] - self.game_state.end_point[0]) <= constants.TERMINAL_CHECK_PADDING and
                abs(self.pose[1] - self.game_state.end_point[1]) <= constants.TERMINAL_CHECK_PADDING):
            self.pose_indicator[
                    self.pose[1] - self.game_state.end_point[1] + constants.TERMINAL_CHECK_PADDING,
                    self.pose[0] - self.game_state.end_point[0] + constants.TERMINAL_CHECK_PADDING] = 1

        self.feed_dict = {
            self.network.image_placeholder: image,
            self.network.action_placeholder: self.action[np.newaxis, np.newaxis, :],
            self.network.pose_placeholder: np.array(self.gt_graph.get_shifted_pose(self.pose))[np.newaxis, np.newaxis, :3],
            self.network.memory_placeholders: self.memory[np.newaxis, ...],
            self.network.gru_placeholder: self.gru_state,
            self.network.pose_indicator_placeholder: self.pose_indicator[np.newaxis, np.newaxis, ...],
            self.network.goal_pose_placeholder: self.goal_pose[np.newaxis, ...],
            }
        if self.num_unrolls is None:
            self.feed_dict[self.network.num_unrolls] = 1

        outputs = self.sess.run(
            [self.network.patch_weights_clipped,
                self.network.gru_state,
                self.network.occupancy,
                self.network.gru_outputs_full,
                self.network.is_possible_sigm,
             ],
            feed_dict=self.feed_dict)

        self.map_weights = outputs[0][0, 0, ...]
        self.game_state.graph.update_graph((self.map_weights, [1 + graph_obj.EPSILON]), self.pose, rows=[0])

        self.gru_state = outputs[1]
        self.occupancy = outputs[2][0, :self.bounds[3], :self.bounds[2], 0] * (self.game_state.graph.memory[:, :, 0] > 1)
        self.memory = outputs[3][0, 0, ...]
        self.is_possible = outputs[4][0, 0]

    def reset(self, scene_name=None, seed=None):
        if scene_name is not None:
            if self.game_state.env is not None and type(self.game_state) == GameState:
                self.game_state.reset(scene_name, use_gt=False, seed=seed)
            self.gt_graph = graph_obj.Graph('layouts/%s-layout.npy' % scene_name, use_gt=True)
            self.bounds = [self.game_state.graph.xMin, self.game_state.graph.yMin,
                self.game_state.graph.xMax - self.game_state.graph.xMin + 1,
                self.game_state.graph.yMax - self.game_state.graph.yMin + 1]
            if len(self.game_state.end_point) == 0:
                self.game_state.end_point = (self.game_state.graph.xMin + constants.TERMINAL_CHECK_PADDING,
                        self.game_state.graph.yMin + constants.TERMINAL_CHECK_PADDING, 0)
            self.action = np.zeros(self.action_util.num_actions)
            self.memory = np.zeros((constants.SPATIAL_MAP_HEIGHT, constants.SPATIAL_MAP_WIDTH, constants.MEMORY_SIZE))
            self.gru_state = np.zeros((1, constants.GRU_SIZE))
            self.pose = self.game_state.pose
            self.is_possible = 1
            self.num_steps = 0
            self.times = np.zeros(2)
            self.impossible_spots = set()
            self.visited_spots = set()
        else:
            self.game_state.reset()

        self.goal_pose = np.array([self.game_state.end_point[0] - self.game_state.graph.xMin,
                self.game_state.end_point[1] - self.game_state.graph.yMin],
                dtype=np.int32)[:2]
        self.inference()

    def step(self, action):
        t_start = time.time()
        if type(self.game_state) == GameState:
            self.game_state.step(action)
        else:
            super(QuestionGameState, self.game_state).step(action)
        self.times[1] += time.time() - t_start
        if self.num_steps % 100 == 0:
            print('game state step time %.3f' % (self.times[1] / (self.num_steps + 1)))
        self.pose = self.game_state.pose
        self.action[:] = 0
        self.action[self.action_util.action_dict_to_ind(action)] = 1
        self.inference()
        self.num_steps += 1
        self.global_step_id += 1

        if not self.game_state.event.metadata['lastActionSuccess']:
            # Can't traverse here, make sure the weight is correct.
            if self.pose[2] == 0:
                self.gt_graph.update_weight(self.pose[0], self.pose[1] + 1, graph_obj.MAX_WEIGHT)
                spot = (self.pose[0], self.pose[1] + 1)
            elif self.pose[2] == 1:
                self.gt_graph.update_weight(self.pose[0] + 1, self.pose[1], graph_obj.MAX_WEIGHT)
                spot = (self.pose[0] + 1, self.pose[1])
            elif self.pose[2] == 2:
                self.gt_graph.update_weight(self.pose[0], self.pose[1] - 1, graph_obj.MAX_WEIGHT)
                spot = (self.pose[0], self.pose[1] - 1)
            elif self.pose[2] == 3:
                self.gt_graph.update_weight(self.pose[0] - 1, self.pose[1], graph_obj.MAX_WEIGHT)
                spot = (self.pose[0] - 1, self.pose[1])
            self.impossible_spots.add(spot)
        else:
            self.visited_spots.add((self.pose[0], self.pose[1]))
        for spot in self.impossible_spots:
            graph_max = self.gt_graph.memory[:, :, 0].max()
            self.game_state.graph.update_weight(spot[0], spot[1], graph_max)
            self.occupancy[spot[1], spot[0]] = 1

    def get_plan(self):
        self.plan, self.path = self.game_state.graph.get_shortest_path(self.pose, self.game_state.end_point)
        return self.plan, self.path

    def get_label(self):
        patch, curr_point = self.gt_graph.get_graph_patch(self.pose)
        patch = patch[:, :, 0]
        patch[patch < 2] = 0
        patch[patch > 1] = 1
        return patch

    def draw_state(self, return_list=False):
        if not constants.DRAWING:
            return
        from utils import drawing
        curr_image = self.game_state.detection_image.copy()
        curr_depth = self.game_state.s_t_depth
        if curr_depth is not None:
            curr_depth = self.game_state.s_t_depth.copy()
            curr_depth[0, 0] = 0
            curr_depth[0, 1] = constants.MAX_DEPTH

        label = np.flipud(self.get_label())
        patch = np.flipud(self.game_state.graph.get_graph_patch(self.pose)[0])
        state_image = self.game_state.draw_state().copy()
        memory_map = np.flipud(self.game_state.graph.memory.copy())
        memory_map = np.concatenate((memory_map[:, :, [0]], np.zeros(memory_map[:, :,[0]].shape), memory_map[:, :, 1:]), axis=2)

        images = [
                curr_image,
                state_image,

                np.minimum(memory_map[:, :, 0], 200),
                np.argmax(memory_map[:, :, 1:], axis=2),

                label[:, :],
                np.minimum(patch[:, :, 0], 10),
                ]
        if return_list:
            return images
        action_str = 'action: %s possible %.3f' % (self.action_util.actions[np.where(self.action == 1)[0].squeeze()]['action'], self.is_possible)
        titles = ['%07d' % self.num_steps, action_str, 'Occupancy Map', 'Objects Map', 'Label Patch', 'Learned Patch']
        image = drawing.subplot(images, 4, 3, curr_image.shape[1], curr_image.shape[0],
                titles=titles, border=3)

        return image


class RLGraphAgent(QAAgent):
    def __init__(self, sess, num_unrolls=1, free_space_network_scope=None, depth_scope=None):
        super(RLGraphAgent, self).__init__(sess, depth_scope)
        self.network = QAPlannerNetwork(constants.RL_GRU_SIZE, 1, num_unrolls)
        self.network.create_net()

        self.num_steps = 0
        self.global_step_id = 0
        self.global_num_steps = 0
        self.num_invalid_actions = 0
        self.global_num_invalid_actions = 0
        self.num_unrolls = num_unrolls
        self.coord_box = np.mgrid[int(-constants.STEPS_AHEAD * 1.0 / 2):np.ceil(constants.STEPS_AHEAD * 1.0 / 2),
                                  1:1 + constants.STEPS_AHEAD].transpose(1, 2, 0) / constants.STEPS_AHEAD

        if constants.USE_NAVIGATION_AGENT:
            self.nav_agent = GraphAgent(sess, True, 1, self.game_state, free_space_network_scope)

    def inference(self):
        outputs = self.get_next_output()
        self.gru_state = outputs[0]
        self.v = outputs[1][0, ...]
        self.pi = outputs[2][0, ...]
        self.possible_moves_pred = outputs[3][0, ...]
        if self.game_state.question_type_ind == 1:
            self.answer = outputs[5][0, ...]
        else:
            self.answer = outputs[4][0, ...]
        self.question = outputs[6]
        self.memory_crops = outputs[7]
        self.possible_moves_weights = outputs[8]
        self.actions = outputs[9]
        self.teleport_input_crops = outputs[10]
        self.pi_logits = outputs[-1]

    def get_next_output(self):
        image = self.game_state.s_t[np.newaxis, np.newaxis, ...]

        pose_shifted = np.array(self.pose[:3])
        pose_shifted[0] -= self.bounds[0]
        pose_shifted[1] -= self.bounds[1]

        self.map_mask_padded = np.pad(self.spatial_map.memory[:, :, 1:],
                ((0, constants.SPATIAL_MAP_HEIGHT - self.bounds[3]),
                (0, constants.SPATIAL_MAP_WIDTH - self.bounds[2]), (0, 0)),
                'constant', constant_values=0).copy()

        self.feed_dict = {
            self.network.image_placeholder: image,
            self.network.pose_placeholder: pose_shifted[np.newaxis, np.newaxis, :],
            self.network.map_mask_placeholder: self.map_mask_padded[np.newaxis, np.newaxis, ...],
            self.network.gru_placeholder: self.gru_state,
            self.network.question_object_placeholder: np.array(self.game_state.object_target)[np.newaxis, np.newaxis],
            self.network.question_container_placeholder: self.game_state.container_target[np.newaxis, np.newaxis, :],
            self.network.question_direction_placeholder: self.game_state.direction_target[np.newaxis, np.newaxis, :],
            self.network.question_type_placeholder: np.array(self.game_state.question_type_ind)[np.newaxis, np.newaxis],
            self.network.existence_answer_placeholder: np.array(self.game_state.answer)[np.newaxis, np.newaxis],
            self.network.counting_answer_placeholder: np.array(self.game_state.answer)[np.newaxis, np.newaxis],
            self.network.answer_weight: np.array(int(self.game_state.can_end))[np.newaxis, np.newaxis],
            self.network.possible_move_placeholder: np.array(self.possible_moves)[np.newaxis, np.newaxis, :],
            self.network.meta_action_placeholder: self.last_meta_action[np.newaxis, np.newaxis, :],
            self.network.taken_action: self.last_action_one_hot[np.newaxis, :],
            self.network.episode_length_placeholder: np.array([self.num_steps / constants.MAX_EPISODE_LENGTH])[np.newaxis, :],
            self.network.question_count_placeholder: np.array([self.question_count])[np.newaxis, :],
            }
        if self.num_unrolls is None:
            self.feed_dict[self.network.num_unrolls] = 1

        self.prev_map_mask = self.spatial_map.memory.copy()

        outputs = self.sess.run(
            [self.network.gru_state, self.network.v, self.network.pi,
                self.network.possible_moves,
                self.network.existence_answer,
                self.network.counting_answer,
                self.network.question_object_one_hot,
                self.network.memory_crops_rot,
                self.network.possible_moves_weights,
                self.network.actions,
                self.network.teleport_input_crops,
                self.network.pi_logits,
                ],
            feed_dict=self.feed_dict)
        return outputs


    def get_reward(self):
        self.reward = self.game_state.reward
        self.reward += self.new_coverage * 1.0 / (constants.STEPS_AHEAD**2 + 1)
        if constants.DEBUG:
            print('coverage %.2f - (%.3f%%)  reward %.3f' % (float(self.coverage), float(self.coverage * 100.0 / self.max_coverage), self.reward))
        if self.game_state.can_end and not self.prev_can_end:
            self.reward = 10
        if self.game_state.can_end and self.prev_can_end:
            if self.game_state.question_type_ind == 0:
                self.reward = -1
            elif self.game_state.question_type_ind == 1:
                pass
            elif self.game_state.question_type_ind == 2:
                self.reward = -1
            elif self.game_state.question_type_ind == 3:
                pass
        if self.terminal:
            if constants.DEBUG:
                print('answering', self.answer)
            if self.game_state.question_type_ind != 1:
                answer = self.answer[0] > 0.5
            else:
                answer = np.argmax(self.answer)

            if answer == self.game_state.answer and self.game_state.can_end:
                self.reward = 10
                print('Answer correct!!!!!')
            else:
                self.reward = -30 # Average is -10 for 50% chance, -20 for 25% chance
                print('Answer incorrect :( :( :( :( :(')
        if self.num_steps >= constants.MAX_EPISODE_LENGTH:
            self.reward = -30
            self.terminal = True
        self.prev_can_end = self.game_state.can_end
        return np.clip(self.reward / 10, -3, 1), self.terminal

    def reset(self, seed=None, test_ind=None):
        if self.game_state.env is not None:
            self.game_state.reset(seed=seed, test_ind=test_ind)
            self.bounds = self.game_state.bounds
        self.end_points = np.zeros_like(self.game_state.graph.memory[:, :, 0])
        for end_point in self.game_state.end_point:
            self.end_points[end_point[1] - self.game_state.graph.yMin, end_point[0] - self.game_state.graph.xMin] = 1

        self.gru_state = np.zeros((1, constants.RL_GRU_SIZE))
        self.pose = self.game_state.pose
        self.prev_pose = self.pose
        self.visited_poses = set()
        self.reward = 0
        self.num_steps = 0
        self.question_count = 0
        self.num_invalid_actions = 0
        self.prev_can_end = False
        dilation_kernel = np.ones((constants.SCENE_PADDING * 2 + 1, constants.SCENE_PADDING))
        free_space = self.game_state.xray_graph.memory.copy().squeeze()
        if len(free_space.shape) == 3:
            free_space = free_space[:, :, 0]
        free_space[free_space == graph_obj.MAX_WEIGHT] = 0
        free_space[free_space > 1] = 1
        self.dilation = np.zeros_like(free_space)
        for _ in range(2):
            self.dilation += scipy.ndimage.morphology.binary_dilation(
                    free_space, structure=dilation_kernel).astype(np.int)
            dilation_kernel = np.rot90(dilation_kernel)
        self.dilation[self.dilation > 1] = 1
        self.max_coverage = np.sum(self.dilation)
        # Rows are:
        # 0 - Map weights (not fed to decision network)
        # 1 and 2 - meshgrid
        # 3 - coverage
        # 4 - teleport locations
        # 5 - free space map
        # 6 - visited locations
        # 7+ - object location
        if constants.USE_NAVIGATION_AGENT:
            if self.nav_agent is not None:
                self.nav_agent.reset(self.game_state.scene_name)
        self.spatial_map = graph_obj.Graph('layouts/%s-layout.npy' % self.game_state.scene_name, use_gt=True, construct_graph=False)
        self.spatial_map.memory = np.concatenate(
                (np.zeros((self.bounds[3], self.bounds[2], 7)),
                    self.game_state.graph.memory[:, :, 1:].copy()), axis=2)

        self.coverage = 0
        self.terminal = False
        self.new_coverage = 0
        self.last_meta_action = np.zeros(7)
        self.last_action_one_hot = np.zeros(self.network.pi.get_shape().as_list()[-1])
        self.global_step_id += 1
        self.update_spatial_map({'action': 'Initialize'})

        # For drawing
        self.forward_pred = np.zeros((3, 3, 3))
        self.next_memory_crops_rot = np.zeros((3, 3, 3))

    def update_spatial_map(self, action):
        if action['action'] == 'Teleport':
            self.spatial_map.memory[action['z'] - self.bounds[1], action['x'] - self.bounds[0], 4] = 1

        self.pose = self.game_state.pose
        self.spatial_map.memory[:, :, 1:3] = 0
        self.spatial_map.update_graph((self.coord_box, 0), self.pose, rows=[1, 2])

        if not constants.USE_NAVIGATION_AGENT:
            path = self.game_state.xray_graph.get_shortest_path(self.prev_pose, self.pose)[1]
            if action['action'] == 'Teleport':
                self.num_steps += max(1, len(path) - 1)

            self.new_coverage = -1
            full_new_coverage = 0
            for pose in path[::-1]:
                patch = self.spatial_map.get_graph_patch(pose)
                patch_coverage = 1 - patch[0][:, :, 3]
                patch_coverage = np.sum(patch_coverage) + (1 - patch[1][3])
                if self.new_coverage == -1:
                    self.new_coverage = patch_coverage
                full_new_coverage += patch_coverage
                self.spatial_map.update_graph((np.ones((constants.STEPS_AHEAD, constants.STEPS_AHEAD, 1)), 1), pose, rows=[3])
                self.spatial_map.memory[pose[1] - self.bounds[1],pose[0] - self.bounds[0], 6] = 1
            self.new_coverage = max(0, self.new_coverage)
            if full_new_coverage > 0:
                # Make sure that new_coverage is positive if coverage increased at all.
                self.new_coverage += 0.00001
            self.coverage += full_new_coverage

            # Do GT occupancy map stuff.
            free_space = self.game_state.xray_graph.memory.copy().squeeze()
            if len(free_space.shape) == 3:
                free_space = free_space[:, :, 0]
            free_space[free_space == graph_obj.MAX_WEIGHT] = 0
            free_space[free_space > 1] = 1
            self.spatial_map.memory[:, :, 5] = free_space * self.spatial_map.memory[:, :, 3]
            #self.game_state.graph.memory[:, :, 0] = (1 + self.spatial_map.memory[:, :, 3] * (graph_obj.EPSILON +
                #(1 - self.spatial_map.memory[:, :, 5]) * (graph_obj.MAX_WEIGHT - graph_obj.EPSILON - 1)))

        else:
            self.spatial_map.memory[:, :, 3] = self.game_state.graph.memory[:, :, 0] > 1
            new_coverage = np.sum(self.spatial_map.memory[:, :, 3])
            self.new_coverage = new_coverage - self.coverage
            self.coverage = new_coverage
            self.spatial_map.memory[:, :, 5] = 1 - self.nav_agent.occupancy
            for pose in self.nav_agent.visited_spots:
                self.spatial_map.memory[pose[1] - self.bounds[1],pose[0] - self.bounds[0], 6] = 1

        if constants.RECORD_FEED_DICT:
            self.spatial_map.memory[:, :, 7:] = self.game_state.xray_graph.memory[:, :, 1:].copy()
        else:
            self.spatial_map.memory[:, :, 7:] = self.game_state.graph.memory[:, :, 1:].copy()

        patch = self.game_state.xray_graph.get_graph_patch(self.pose)[0].reshape((constants.STEPS_AHEAD ** 2, -1))
        patch = patch[:, 0]
        patch[patch == graph_obj.MAX_WEIGHT] = 0
        patch[patch > 1] = 1
        open_success = not self.game_state.get_action({'action': 'OpenObject'})[2]
        close_success = not self.game_state.get_action({'action': 'CloseObject'})[2]
        self.possible_moves = np.concatenate((patch, [1], [1],
            [self.pose[3] != 330], [self.pose[3] != 60], [open_success], [close_success]))

    def step(self, action):
        self.prev_pose = self.pose
        self.visited_poses.add(self.pose)

        self.last_meta_action = np.zeros(7)
        if action['action'] == 'Teleport':
            self.last_meta_action[0] = 1
        elif action['action'] == 'RotateLeft':
            self.last_meta_action[1] = 1
        elif action['action'] == 'RotateRight':
            self.last_meta_action[2] = 1
        elif action['action'] == 'LookUp':
            self.last_meta_action[3] = 1
        elif action['action'] == 'LookDown':
            self.last_meta_action[4] = 1
        elif action['action'] == 'OpenObject':
            self.last_meta_action[5] = 1
        elif action['action'] == 'CloseObject':
            self.last_meta_action[6] = 1

        if action['action'] == 'Answer':
            self.terminal = True
        else:
            if not constants.USE_NAVIGATION_AGENT or action['action'] != 'Teleport':
                self.game_state.step(action)
                if not self.game_state.event.metadata['lastActionSuccess']:
                    self.num_invalid_actions += 1
                    self.global_num_invalid_actions += 1
                if action['action'] != 'Teleport':
                    self.num_steps += 1
                    self.global_num_steps += 1
            else:
                num_steps, num_invalid_actions = self.nav_agent.goto(action, self.global_step_id)
                # Still need to step to get reward etc.
                self.global_num_steps += num_steps
                self.global_step_id += num_steps
                self.num_steps += num_steps
                self.game_state.step(action)
                self.num_invalid_actions += num_invalid_actions
                self.global_num_invalid_actions += num_invalid_actions
            if constants.USE_NAVIGATION_AGENT and 'Rotate' in action['action']:
                self.nav_agent.inference()

            self.update_spatial_map(action)

        self.global_step_id += 1

    def get_action(self, action_ind):
        if action_ind < constants.STEPS_AHEAD ** 2:
            # Teleport
            action_x = action_ind % constants.STEPS_AHEAD - int(constants.STEPS_AHEAD / 2)
            action_z = int(action_ind / constants.STEPS_AHEAD) + 1
            x_shift = 0
            z_shift = 0
            if self.pose[2] == 0:
                x_shift = action_x
                z_shift = action_z
            elif self.pose[2] == 1:
                x_shift = action_z
                z_shift = -action_x
            elif self.pose[2] == 2:
                x_shift = -action_x
                z_shift = -action_z
            elif self.pose[2] == 3:
                x_shift = -action_z
                z_shift = action_x
            action_x = self.pose[0] + x_shift
            action_z = self.pose[1] + z_shift
            action = {
                'action': 'Teleport',
                'x': action_x,
                'z': action_z,
                'rotation': self.pose[2] * 90,
                }
        else:
            # Rotate/Look/Open/Close/Answer
            action_ind -= constants.STEPS_AHEAD ** 2
            if action_ind == 0:
                action = {'action': 'RotateLeft'}
            elif action_ind == 1:
                action = {'action': 'RotateRight'}
            elif action_ind == 2:
                action = {'action': 'LookUp'}
            elif action_ind == 3:
                action = {'action': 'LookDown'}
            elif action_ind == 4:
                action = {'action': 'OpenObject'}
            elif action_ind == 5:
                action = {'action': 'CloseObject'}
            elif action_ind == 6:
                action = {'action': 'Answer', 'value': self.answer}
            else:
                raise Exception('something very wrong happened')
        return action

    def draw_state(self, return_list=False, action=None):
        if not constants.DRAWING:
            return
        # Rows are:
        # 0 - Map weights (not fed to decision network)
        # 1 and 2 - meshgrid
        # 3 - coverage
        # 4 - teleport locations
        # 5 - free space map
        # 6 - visited locations
        # 7+ - object location
        from utils import drawing
        curr_image = self.game_state.detection_image.copy()
        state_image = self.game_state.draw_state()

        action_hist = np.zeros((3, 3, 3))
        pi = self.pi.copy()
        if constants.STEPS_AHEAD == 5:
            action_hist = np.concatenate((pi, np.zeros(3)))
            action_hist = action_hist.reshape(7, 5)
        elif constants.STEPS_AHEAD == 1:
            action_hist = np.concatenate((pi, np.zeros(1)))
            action_hist = action_hist.reshape(3, 3)

        flat_action_size = max(len(pi), 100)
        flat_action_hist = np.zeros((flat_action_size, flat_action_size))
        for ii,flat_action_i in enumerate(pi):
            flat_action_hist[:max(int(np.round(flat_action_i * flat_action_size)), 1),
                    int(ii * flat_action_size / len(pi)):int((ii+1) * flat_action_size / len(pi))] = (ii + 1)
        flat_action_hist = np.flipud(flat_action_hist)

        # Answer histogram
        ans = self.answer
        if len(ans) == 1:
            ans = [1 - ans[0], ans[0]]
        ans_size = max(len(ans), 100)
        ans_hist = np.zeros((ans_size, ans_size))
        for ii,ans_i in enumerate(ans):
            ans_hist[:max(int(np.round(ans_i * ans_size)), 1),
                    int(ii * ans_size / len(ans)):int((ii+1) * ans_size / len(ans))] = (ii + 1)
        ans_hist = np.flipud(ans_hist)

        dil = np.flipud(self.dilation)
        dil[0, 0] = 4
        coverage = int(self.coverage * 100 / self.max_coverage)

        possible = np.zeros((3, 3, 3))
        possible_pred = np.zeros((3, 3, 3))
        if constants.STEPS_AHEAD == 5:
            possible = self.possible_moves.copy()
            possible = np.concatenate((possible, np.zeros(4)))
            possible = possible.reshape(constants.STEPS_AHEAD + 2, constants.STEPS_AHEAD)

            possible_pred = self.possible_moves_pred.copy()
            possible_pred = np.concatenate((possible_pred, np.zeros(4)))
            possible_pred = possible_pred.reshape(constants.STEPS_AHEAD + 2, constants.STEPS_AHEAD)

        elif constants.STEPS_AHEAD == 1:
            possible = self.possible_moves.copy()
            possible = np.concatenate((possible, np.zeros(2)))
            possible = possible.reshape(3, 3)

            possible_pred = self.possible_moves_pred.copy()
            possible_pred = np.concatenate((possible_pred, np.zeros(2)))
            possible_pred = possible_pred.reshape(3, 3)


        if self.game_state.question_type_ind in {2, 3}:
            obj_mem = self.spatial_map.memory[:, :, 7 + self.game_state.question_target[1]].copy()
            obj_mem += self.spatial_map.memory[:, :, 7 + self.game_state.object_target] * 2
        else:
            obj_mem = self.spatial_map.memory[:, :, 7 + self.game_state.object_target].copy()
        obj_mem[0, 0] = 2

        memory_map = np.flipud(self.spatial_map.memory[:, :, 7:].copy())
        curr_objs = np.argmax(memory_map, axis=2)

        gt_objs = np.flipud(np.argmax(self.game_state.xray_graph.memory[:, :, 1:], 2))
        curr_objs[0, 0] = np.max(gt_objs)
        memory_crop = self.memory_crops[0, ...].copy()
        memory_crop_cov = np.argmax(np.flipud(memory_crop), axis=2)

        gt_semantic_crop = np.flipud(np.argmax(self.next_memory_crops_rot, axis=2))

        images = [
                curr_image,
                state_image,
                dil + np.max(np.flipud(self.spatial_map.memory[:, :, 3:5]) * np.array([1, 3]), axis=2),
                memory_crop_cov,

                ans_hist,
                flat_action_hist,
                np.flipud(action_hist),
                np.flipud(possible),

                np.flipud(possible_pred),
                gt_objs,
                curr_objs,
                np.flipud(obj_mem),
                ]
        if type(action) == int:
            action = self.game_state.get_action(action)[0]
        action_str = game_util.get_action_str(action)
        if action_str == 'Answer':
            if self.game_state.question_type_ind != 1:
                action_str += ' ' + str(self.answer > 0.5)
            elif self.game_state.question_type_ind == 1:
                action_str += ' ' + str(np.argmax(self.answer))
        if self.game_state.question_type_ind == 0:
            question_str = '%03d S %s Ex Q: %s A: %s' % (
                    self.num_steps, self.game_state.scene_name[9:],
                    constants.OBJECTS[self.game_state.question_target], bool(self.game_state.answer))
        elif self.game_state.question_type_ind == 1:
            question_str = '%03d S %s # Q: %s A: %d' % (
                    self.num_steps, self.game_state.scene_name[9:],
                    constants.OBJECTS[self.game_state.question_target], self.game_state.answer)
        elif self.game_state.question_type_ind == 2:
            question_str = '%03d S %s Q: %s in %s A: %s' % (
                    self.num_steps,
                    self.game_state.scene_name[9:],
                    constants.OBJECTS[self.game_state.question_target[0]],
                    constants.OBJECTS[self.game_state.question_target[1]],
                    bool(self.game_state.answer))
        else:
            raise Exception('No matching question number')
        titles=[
            question_str,
            str(self.answer),
            action_str,
            'coverage %d%% can end %s' % (coverage, bool(self.game_state.can_end)),
            'reward %.3f, value %.3f' % (self.reward, self.v),
            ]
        if return_list:
            return action_hist
        image = drawing.subplot(images, 4, 3, curr_image.shape[1], curr_image.shape[0], titles=titles, border=3)
        if not os.path.exists('visualizations/images'):
            os.makedirs('visualizations/images')
        cv2.imwrite('visualizations/images/state_%05d.jpg' % self.global_step_id, image[:, :, ::-1])

        return image

