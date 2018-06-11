import numpy as np
import os
import cv2
from utils import game_util

import constants

from networks.end_to_end_baseline_network import EndToEndBaselineNetwork
from qa_agents.qa_agent import QAAgent


class EndToEndBaselineGraphAgent(QAAgent):
    def __init__(self, sess, num_unrolls=1, depth_scope=None):
        super(EndToEndBaselineGraphAgent, self).__init__(sess, depth_scope)
        self.network = EndToEndBaselineNetwork()
        self.network.create_net()

        self.num_steps = 0
        self.global_num_steps = 0
        self.global_step_id = 0
        self.num_invalid_actions = 0
        self.global_num_invalid_actions = 0
        self.num_unrolls = num_unrolls
        self.coord_box = np.mgrid[int(-constants.STEPS_AHEAD * 1.0 / 2):np.ceil(constants.STEPS_AHEAD * 1.0 / 2), 1:1 + constants.STEPS_AHEAD].transpose(1, 2, 0) / constants.STEPS_AHEAD

    def reset(self, seed=None, test_ind=None):
        if self.game_state.env is not None:
            self.game_state.reset(seed=seed, test_ind=test_ind)
            self.bounds = self.game_state.bounds
        self.gru_state = np.zeros((1, constants.RL_GRU_SIZE))
        self.pose = self.game_state.pose
        self.prev_pose = self.pose
        self.visited_poses = set()
        self.reward = 0
        self.num_steps = 0
        self.question_count = 0
        self.num_invalid_actions = 0

        self.prev_can_end = False
        self.terminal = False
        self.last_meta_action = np.zeros(7)
        self.last_action_one_hot = np.zeros(self.network.pi.get_shape().as_list()[-1])
        self.global_step_id += 1

    def inference(self):
        outputs = self.get_next_output()
        self.gru_state = outputs[0]
        self.v = outputs[1][0, ...]
        self.pi = outputs[2][0, ...]
        if self.game_state.question_type_ind == 1:
            self.answer = outputs[4][0, ...]
        else:
            self.answer = outputs[3][0, ...]

    def get_next_output(self):
        detection_image = self.game_state.detection_mask_image
        image = self.game_state.s_t[np.newaxis, np.newaxis, ...]

        self.feed_dict = {
            self.network.detection_image: detection_image[np.newaxis, np.newaxis, ...],
            self.network.image_placeholder: image,
            self.network.gru_placeholder: self.gru_state,
            self.network.question_object_placeholder: np.array(self.game_state.object_target)[np.newaxis, np.newaxis],
            self.network.question_container_placeholder: self.game_state.container_target[np.newaxis, np.newaxis, :],
            self.network.question_direction_placeholder: self.game_state.direction_target[np.newaxis, np.newaxis, :],
            self.network.question_type_placeholder: np.array(self.game_state.question_type_ind)[np.newaxis, np.newaxis],
            self.network.existence_answer_placeholder: np.array(self.game_state.answer)[np.newaxis, np.newaxis],
            self.network.counting_answer_placeholder: np.array(self.game_state.answer)[np.newaxis, np.newaxis],
            self.network.action_placeholder: self.last_meta_action[np.newaxis, np.newaxis, :],
            self.network.answer_weight: np.array(int(self.game_state.can_end))[np.newaxis, np.newaxis],
        }
        if self.num_unrolls is None:
            self.feed_dict[self.network.num_unrolls] = 1

        outputs = self.sess.run(
            [self.network.gru_state, self.network.v, self.network.pi,
             self.network.existence_answer,
             self.network.counting_answer,
             ],
            feed_dict=self.feed_dict)
        return outputs

    def get_reward(self):
        self.reward = self.game_state.reward

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
                self.reward = -30  # Average is -10 for 50% chance, -20 for 25% chance
                print('Answer incorrect :( :( :( :( :(')
        if self.num_steps >= constants.MAX_EPISODE_LENGTH:
            self.reward = -30
            self.terminal = True
        self.prev_can_end = self.game_state.can_end
        return np.clip(self.reward / 10, -3, 1), self.terminal

    def get_action(self, action_ind):
        if action_ind == 0:
            action = {'action': 'MoveAhead', 'moveMagnitude': constants.AGENT_STEP_SIZE}
        elif action_ind == 1:
            action = {'action': 'RotateLeft'}
        elif action_ind == 2:
            action = {'action': 'RotateRight'}
        elif action_ind == 3:
            action = {'action': 'LookUp'}
        elif action_ind == 4:
            action = {'action': 'LookDown'}
        elif action_ind == 5:
            action = {'action': 'OpenObject'}
        elif action_ind == 6:
            action = {'action': 'CloseObject'}
        elif action_ind == 7:
            action = {'action': 'Answer', 'value': self.answer}
        else:
            raise Exception('something very wrong happened')
        return action

    def step(self, action):
        self.visited_poses.add(self.pose)

        self.last_meta_action = np.zeros(7)
        if action['action'] == 'MoveAhead':
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
            self.game_state.step(action)
            self.pose = self.game_state.pose

        self.num_invalid_actions += 1 - int(self.game_state.event.metadata['lastActionSuccess'])
        self.global_num_invalid_actions += 1 - int(self.game_state.event.metadata['lastActionSuccess'])
        self.num_steps += 1
        self.global_num_steps += 1
        self.global_step_id += 1

    def draw_state(self, return_list=False, action=None):
        if not constants.DRAWING:
            return
        from utils import drawing
        curr_image = self.game_state.detection_image.copy()
        state_image = self.game_state.draw_state()
        pi = self.pi.copy().squeeze()

        action_size = max(len(pi), 100)
        action_hist = np.zeros((action_size, action_size))
        for ii,pi_i in enumerate(pi):
            action_hist[:max(int(np.round(pi_i * action_size)), 1),
            int(ii * action_size / len(pi)):int((ii+1) * action_size / len(pi))] = (ii + 1)
        action_hist = np.flipud(action_hist)
        images = [
            curr_image,
            np.argmax(self.game_state.detection_mask_image, 2),
            action_hist,
            state_image,
        ]
        if type(action) == int:
            action = self.game_state.get_action(action)[0]
        action_str = game_util.get_action_str(action)
        if action_str == 'Answer':
            if self.game_state.question_type_ind != 1:
                action_str += ' ' + str(self.answer > 0.5)
            else:
                action_str += ' ' + str(np.argmax(self.answer))

        if self.game_state.question_type_ind == 0:
            question_str = '%03d Ex Q: %s A: %s' % (self.num_steps, constants.OBJECTS[self.game_state.question_target], bool(self.game_state.answer))
        elif self.game_state.question_type_ind == 1:
            question_str = '%03d # Q: %s A: %d' % (self.num_steps, constants.OBJECTS[self.game_state.question_target], self.game_state.answer)
        elif self.game_state.question_type_ind == 2:
            question_str = '%03d Q: %s in %s A: %d' % (self.num_steps,
                                                       constants.OBJECTS[self.game_state.question_target[0]],
                                                       constants.OBJECTS[self.game_state.question_target[1]],
                                                       self.game_state.answer)
        else:
            raise Exception('No matching question number')

        titles=[question_str,
                action_str,
                'reward %.3f, value %.3f' % (self.reward, self.v)]
        if return_list:
            return action_hist
        image = drawing.subplot(images, 2, 2, curr_image.shape[1], curr_image.shape[0], titles=titles)
        if not os.path.exists('visualizations/images'):
            os.makedirs('visualizations/images')
        cv2.imwrite('visualizations/images/state_%05d.jpg' % self.global_step_id, image[:, :, ::-1])
        return image

