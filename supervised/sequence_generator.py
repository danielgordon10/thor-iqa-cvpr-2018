import numpy as np
import random
from utils import drawing
from qa_agents import graph_agent

import constants
from utils import game_util

class SequenceGenerator(object):
    def __init__(self, sess):
        self.agent = graph_agent.GraphAgent(sess, reuse=True)
        self.game_state = self.agent.game_state
        self.action_util = self.game_state.action_util
        self.planner_prob = 0.5
        self.scene_num = 0
        self.count = -1
        self.scene_name = None

    def generate_episode(self):
        self.count += 1
        self.states = []
        self.debug_images = []
        planning = random.random() < self.planner_prob

        while len(self.states) == 0:
            if self.count % 5 == 0:
                self.scene_name = 'FloorPlan%d' % random.choice(constants.SCENE_NUMBERS)
                print('New episode. Scene %s' % self.scene_name)
                self.agent.reset(self.scene_name)
            else:
                self.agent.reset()

            label = self.agent.get_label()
            pose = game_util.get_pose(self.game_state.event)[:3]

            if constants.DRAWING:
                patch = self.game_state.graph.get_graph_patch(pose)[0]
                self.debug_images.append({
                    'color': self.game_state.s_t,
                    'label': np.flipud(label),
                    'patch': np.flipud(patch),
                    'label_memory': np.minimum(np.flipud(self.agent.gt_graph.memory.copy()), 2),
                    'state_image': self.game_state.draw_state().copy(),
                    'memory_map': np.minimum(np.flipud(self.game_state.graph.memory.copy()), 10),
                    'pose_indicator': np.flipud(self.agent.pose_indicator),
                    'detections': self.game_state.detection_image if constants.OBJECT_DETECTION else None,
                    'weight': 1,
                    'possible_label': (1 if self.game_state.graph.memory[
                                                self.game_state.end_point[1] - self.game_state.graph.yMin,
                                                self.game_state.end_point[0] - self.game_state.graph.xMin, 0] == 1
                                       else self.game_state.is_possible_end_point),
                    'possible_pred': self.agent.is_possible,
                    })
            self.states.append({
                'color': self.game_state.s_t,
                'pose': self.agent.gt_graph.get_shifted_pose(self.agent.pose)[:3],
                'label': label,
                'action': np.zeros(self.action_util.num_actions),
                'pose_indicator': self.agent.pose_indicator,
                'weight': 1,
                'possible_label': (1 if self.game_state.graph.memory[
                                            self.game_state.end_point[1] - self.game_state.graph.yMin,
                                            self.game_state.end_point[0] - self.game_state.graph.xMin, 0] == 1
                                   else self.game_state.is_possible_end_point),
                })
            optimal_plan, optimal_path = self.agent.gt_graph.get_shortest_path(
                    pose, self.game_state.end_point)
            if planning:
                plan, path = self.agent.get_plan()
            else:
                plan = optimal_plan
                path = optimal_path
            num_iters = 0
            seen_terminal = False
            while ((not seen_terminal) and len(plan) != 0 and
                    self.agent.is_possible >= constants.POSSIBLE_THRESH):
                num_iters += 1
                if constants.DEBUG:
                    print('num iters', num_iters, 'max', constants.MAX_EPISODE_LENGTH)
                if num_iters > constants.MAX_EPISODE_LENGTH:
                    print('Path length too long in scene',
                          self.scene_name, 'goal_position', self.game_state.end_point,
                          'pose', pose, 'plan', plan)
                    plan = []
                    break

                action_vec = np.zeros(self.action_util.num_actions)
                if len(plan) > 0:
                    action = plan[0]
                    self.agent.step(action)
                    action_vec[self.action_util.action_dict_to_ind(action)] = 1
                pose = game_util.get_pose(self.game_state.event)[:3]

                optimal_plan, optimal_path = self.agent.gt_graph.get_shortest_path(
                        pose, self.game_state.end_point)
                if planning:
                    plan, path = self.agent.get_plan()
                else:
                    plan = optimal_plan
                    path = optimal_path

                label = self.agent.get_label()
                self.states.append({
                    'color': self.game_state.s_t,
                    'pose': self.agent.gt_graph.get_shifted_pose(self.agent.pose)[:3],
                    'label': label,
                    'action': action_vec,
                    'pose_indicator': self.agent.pose_indicator,
                    'weight': 1,
                    'possible_label': (1 if self.game_state.graph.memory[
                                               self.game_state.end_point[1] - self.game_state.graph.yMin,
                                               self.game_state.end_point[0] - self.game_state.graph.xMin, 0] == 1
                                       else self.game_state.is_possible_end_point),
                    })
                seen_terminal = seen_terminal or int(len(optimal_plan) == 0)
                if self.states[-1]['label'].shape != (constants.STEPS_AHEAD, constants.STEPS_AHEAD):
                    self.states = []
                    print('Label is wrong size scene', self.scene_name, 'pose', pose)
                    break
                if constants.DRAWING:
                    patch = self.game_state.graph.get_graph_patch(pose)[0]
                    self.debug_images.append({
                        'color': self.game_state.s_t,
                        'label': np.flipud(label),
                        'patch': np.flipud(patch),
                        'label_memory': np.minimum(np.flipud(self.agent.gt_graph.memory.copy()), 2),
                        'state_image': self.game_state.draw_state().copy(),
                        'pose_indicator': np.flipud(self.agent.pose_indicator),
                        'detections': self.game_state.detection_image if constants.OBJECT_DETECTION else None,
                        'memory_map': np.minimum(np.flipud(self.game_state.graph.memory.copy()), 10),
                        'possible_label': (1 if self.game_state.graph.memory[
                                                    self.game_state.end_point[1] - self.game_state.graph.yMin,
                                                    self.game_state.end_point[0] - self.game_state.graph.xMin, 0] == 1
                                           else self.game_state.is_possible_end_point),
                        'possible_pred': self.agent.is_possible,
                        })
        self.bounds = [self.game_state.graph.xMin, self.game_state.graph.yMin,
            self.game_state.graph.xMax - self.game_state.graph.xMin + 1,
            self.game_state.graph.yMax - self.game_state.graph.yMin + 1]
        goal_pose = np.array([self.game_state.end_point[0] - self.game_state.graph.xMin,
                self.game_state.end_point[1] - self.game_state.graph.yMin],
                dtype=np.int32)[:2]
        return (self.states, self.bounds, goal_pose)

if __name__ == '__main__':
    from networks.free_space_network import FreeSpaceNetwork
    from utils import tf_util
    import tensorflow as tf
    sess = tf_util.Session()

    with tf.variable_scope('nav_global_network'):
        network = FreeSpaceNetwork(constants.GRU_SIZE, 1, 1)
        network.create_net()
    sess.run(tf.global_variables_initializer())
    start_it = tf_util.restore_from_dir(sess, constants.CHECKPOINT_DIR)

    import cv2

    sequence_generator = SequenceGenerator(sess)
    sequence_generator.planner_prob = 1
    counter = 0
    while True:
        states, bounds, goal_pose = sequence_generator.generate_episode()
        images = sequence_generator.debug_images
        for im_dict in images:
            counter += 1

            gt_map = (2 - im_dict['label_memory'][:,:,0])

            image_list = [
                    im_dict['detections'] if constants.OBJECT_DETECTION else im_dict['color'],
                    im_dict['state_image'],
                    im_dict['memory_map'][:,:,0],
                    gt_map + np.argmax(im_dict['memory_map'][:,:,1:constants.NUM_RECEPTACLES + 2], axis=2),
                    gt_map + np.argmax(im_dict['memory_map'][:,:,constants.NUM_RECEPTACLES + 2:], axis=2),
                    ]
            titles = ['color', 'state', 'occupied', 'label receptacles', 'label objects']
            print('possible pred', im_dict['possible_pred'])
            image = drawing.subplot(image_list, 2, 2, constants.SCREEN_WIDTH, constants.SCREEN_HEIGHT,
                    titles=titles)
            cv2.imshow('image', image[:,:,::-1])
            cv2.waitKey(0)



