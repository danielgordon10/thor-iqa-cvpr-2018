import pdb
import copy
import time
import random
import numpy as np
import h5py
import glob
import os
from graph import graph_obj
from utils import game_util
from utils import action_util
from utils import bb_util
import tensorflow as tf
from darknet_object_detection import detector

import constants

assert(constants.SCENE_PADDING == 5)


class GameState(object):
    def __init__(self, sess=None, depth_scope=None):
        self.env = game_util.create_env()
        self.action_util = action_util.ActionUtil()
        self.local_random = random.Random()
        if constants.PREDICT_DEPTH:
            from depth_estimation_network import depth_estimator
            if depth_scope is not None:
                with tf.variable_scope(depth_scope, reuse=True):
                    self.depth_estimator = depth_estimator.get_depth_estimator(sess)
            else:
                self.depth_estimator = depth_estimator.get_depth_estimator(sess)
        if constants.OBJECT_DETECTION:
            self.object_detector = detector.get_detector()

        self.im_count = 0
        self.times = np.zeros((4, 2))

    def process_frame(self, run_object_detection=False):
        self.im_count += 1
        self.pose = game_util.get_pose(self.event)

        self.s_t_orig = self.event.frame
        self.s_t = game_util.imresize(self.event.frame, (constants.SCREEN_HEIGHT, constants.SCREEN_WIDTH), rescale=False)
        if constants.DRAWING:
            self.detection_image = self.s_t_orig.copy()
        if constants.PREDICT_DEPTH:
            t_start = time.time()
            self.s_t_depth = self.depth_estimator.get_depth(self.s_t)
            self.times[0, 0] += time.time() - t_start
            self.times[0, 1] += 1
            if self.times[0, 1] % 100 == 0:
                print('depth time  %.3f' % (self.times[0, 0] / self.times[0, 1]))
        elif constants.RENDER_DEPTH_IMAGE:
            self.s_t_depth = game_util.imresize(self.event.frame_depth, (constants.SCREEN_HEIGHT, constants.SCREEN_WIDTH), rescale=False)

        if (constants.GT_OBJECT_DETECTION or constants.OBJECT_DETECTION or
                (constants.END_TO_END_BASELINE and constants.USE_OBJECT_DETECTION_AS_INPUT) and
                not run_object_detection):
            if constants.OBJECT_DETECTION and not run_object_detection:
                # Get detections.

                t_start = time.time()
                boxes, scores, class_names = self.object_detector.detect(game_util.imresize(self.event.frame, (608, 608), rescale=False))
                self.times[1, 0] += time.time() - t_start
                self.times[1, 1] += 1
                if self.times[1, 1] % 100 == 0:
                    print('detection time %.3f' % (self.times[1, 0] / self.times[1, 1]))
                mask_dict = {}
                used_inds = []
                inds = list(range(len(boxes)))
                for (ii, box, score, class_name) in zip(inds, boxes, scores, class_names):
                    if class_name in constants.OBJECT_CLASS_TO_ID:
                        if class_name not in mask_dict:
                            mask_dict[class_name] = np.zeros((constants.SCREEN_HEIGHT, constants.SCREEN_WIDTH), dtype=np.float32)
                        mask_dict[class_name][box[1]:box[3] + 1, box[0]:box[2] + 1] += score
                        used_inds.append(ii)
                mask_dict = {k : np.minimum(v, 1) for k,v in mask_dict.items()}
                used_inds = np.array(used_inds)
                if len(used_inds) > 0:
                    boxes = boxes[used_inds]
                    scores = scores[used_inds]
                    class_names = class_names[used_inds]
                else:
                    boxes = np.zeros((0, 4))
                    scores = np.zeros(0)
                    class_names = np.zeros(0)
                masks = [mask_dict[class_name] for class_name in class_names]

                if constants.END_TO_END_BASELINE:
                    self.detection_mask_image = np.zeros((constants.SCREEN_HEIGHT, constants.SCREEN_WIDTH, len(constants.OBJECTS)), dtype=np.float32)
                    for cls in constants.OBJECTS:
                        if cls not in mask_dict:
                            continue
                        self.detection_mask_image[:, :, constants.OBJECT_CLASS_TO_ID[cls]] = mask_dict[cls]

            else:
                scores = []
                class_names = []
                masks = []
                for (k, v) in self.event.class_masks.items():
                    if k in constants.OBJECT_CLASS_TO_ID and len(v) > 0:
                        scores.append(1)
                        class_names.append(k)
                        masks.append(v)

                if constants.END_TO_END_BASELINE:
                    self.detection_mask_image = np.zeros((constants.SCREEN_HEIGHT, constants.SCREEN_WIDTH, constants.NUM_CLASSES), dtype=np.uint8)
                    for cls in constants.OBJECTS:
                        if cls not in self.event.class_detections2D:
                            continue
                        for box in self.event.class_detections2D[cls]:
                            self.detection_mask_image[box[1]:box[3] + 1, box[0]:box[2] + 1, constants.OBJECT_CLASS_TO_ID[cls]] = 1

            if constants.RENDER_DEPTH_IMAGE or constants.PREDICT_DEPTH:
                xzy = game_util.depth_to_world_coordinates(self.s_t_depth, self.pose, self.camera_height / constants.AGENT_STEP_SIZE)
                max_depth_mask = self.s_t_depth >= constants.MAX_DEPTH
                for ii in range(len(masks)):
                    mask = masks[ii]
                    mask_locs = (mask > 0)
                    locations = xzy[mask_locs, :2]
                    max_depth_locs = max_depth_mask[mask_locs]
                    depth_locs = np.logical_not(max_depth_locs)
                    locations = locations[depth_locs]
                    score = mask[mask_locs]
                    score = score[depth_locs]
                    # remove outliers:
                    locations = locations.reshape(-1, 2)

                    locations = np.round(locations).astype(np.int32)
                    locations -= np.array(self.bounds)[[0, 1]]
                    locations[:, 0] = np.clip(locations[:, 0], 0, self.bounds[2] - 1)
                    locations[:, 1] = np.clip(locations[:, 1], 0, self.bounds[3] - 1)
                    locations, unique_inds = game_util.unique_rows(locations, return_index=True)
                    score = score[unique_inds]

                    curr_score = self.graph.memory[locations[:, 1], locations[:, 0], constants.OBJECT_CLASS_TO_ID[class_names[ii]] + 1]

                    avg_locs = np.logical_and(curr_score > 0, curr_score < 1)
                    curr_score[avg_locs] = curr_score[avg_locs] * .5 + score[avg_locs] * .5
                    curr_score[curr_score == 0] = score[curr_score == 0]
                    self.graph.memory[locations[:, 1], locations[:, 0], constants.OBJECT_CLASS_TO_ID[class_names[ii]] + 1] = curr_score

                    # inverse marked as empty
                    locations = xzy[np.logical_not(mask_locs), :2]
                    max_depth_locs = max_depth_mask[np.logical_not(mask_locs)]
                    depth_locs = np.logical_not(max_depth_locs)
                    locations = locations[depth_locs]
                    locations = locations.reshape(-1, 2)
                    locations = np.round(locations).astype(np.int32)
                    locations[:, 0] = np.clip(locations[:, 0], self.bounds[0], self.bounds[0] + self.bounds[2] - 1)
                    locations[:, 1] = np.clip(locations[:, 1], self.bounds[1], self.bounds[1] + self.bounds[3] - 1)
                    locations = game_util.unique_rows(locations)
                    locations -= np.array(self.bounds)[[0, 1]]
                    curr_score = self.graph.memory[locations[:, 1], locations[:, 0],
                        constants.OBJECT_CLASS_TO_ID[class_names[ii]] + 1]
                    replace_locs = np.logical_and(curr_score > 0, curr_score < 1)
                    curr_score[replace_locs] = curr_score[replace_locs] * .8
                    self.graph.memory[locations[:, 1], locations[:, 0],
                        constants.OBJECT_CLASS_TO_ID[class_names[ii]] + 1] = curr_score
            if constants.DRAWING:
                if constants.GT_OBJECT_DETECTION:
                    boxes = []
                    scores = []
                    class_names = []
                    for k,v in self.event.class_detections2D.items():
                        if k in constants.OBJECT_CLASS_TO_ID and len(v) > 0:
                            boxes.extend(v)
                            scores.extend([1] * len(v))
                            class_names.extend([k] * len(v))
                boxes = np.array(boxes)
                scores = np.array(scores)
                self.detection_image = detector.visualize_detections(self.event.frame, boxes, class_names, scores)

    def reset(self, scene_name=None, use_gt=True, seed=None):
        if scene_name is None:
            # Do half reset
            action_ind = self.local_random.randint(0, constants.STEPS_AHEAD ** 2 - 1)
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
            self.end_point = (action_x, action_z, self.pose[2])

        else:
            # Do full reset
            self.scene_name = scene_name
            grid_file = 'layouts/%s-layout.npy' % scene_name
            self.graph = graph_obj.Graph(grid_file, use_gt=use_gt)
            if seed is not None:
                self.local_random.seed(seed)
            lastActionSuccess = False

            self.bounds = [self.graph.xMin, self.graph.yMin,
                self.graph.xMax - self.graph.xMin + 1,
                self.graph.yMax - self.graph.yMin + 1]

            while not lastActionSuccess:
                self.event = game_util.reset(self.env, self.scene_name)
                self.agent_height = self.event.metadata['agent']['position']['y']
                self.camera_height = self.agent_height + constants.CAMERA_HEIGHT_OFFSET
                self.event = self.env.random_initialize(seed)
                start_point = self.local_random.randint(0, self.graph.points.shape[0] - 1)
                start_point = self.graph.points[start_point, :].copy()
                self.start_point = (start_point[0], start_point[1], self.local_random.randint(0, 3))
                self.end_point = self.start_point
                while self.end_point[0] == self.start_point[0] and self.end_point[1] == self.start_point[1]:
                    end_point = self.local_random.randint(0, self.graph.points.shape[0] - 1)
                    end_point = self.graph.points[end_point, :].copy()
                    self.end_point = [end_point[0], end_point[1], self.local_random.randint(0, 3)]
                    self.end_point[0] += self.local_random.randint(-constants.TERMINAL_CHECK_PADDING, constants.TERMINAL_CHECK_PADDING)
                    self.end_point[1] += self.local_random.randint(-constants.TERMINAL_CHECK_PADDING, constants.TERMINAL_CHECK_PADDING)
                    self.end_point = tuple(self.end_point)

                action = {'action': 'TeleportFull',
                    'x': self.start_point[0] * constants.AGENT_STEP_SIZE,
                    'y': self.agent_height,
                    'z': self.start_point[1] * constants.AGENT_STEP_SIZE,
                    'rotateOnTeleport': True,
                    'rotation': self.start_point[2] * 90,
                    'horizon': 60}
                self.event = self.env.step(action)
                lastActionSuccess = self.event.metadata['lastActionSuccess']

        self.process_frame()
        self.board = None
        point_dists = np.sum(np.abs(self.graph.points - np.array(self.end_point[:2])), axis=1)
        dist_min = np.min(point_dists)
        self.is_possible_end_point = int(dist_min < 0.0001)

    def step(self, action_or_ind):
        if type(action_or_ind) == int:
            action = self.action_util.actions[action_or_ind]
        else:
            action = action_or_ind
        t_start = time.time()

        # The object nearest the center of the screen is open/closed if none is provided.
        if (action['action'] == 'OpenObject' or action['action'] == 'CloseObject') and 'objectId' not in action:
            game_util.set_open_close_object(action, self.event)
        self.event = self.env.step(action)
        self.times[2, 0] += time.time() - t_start
        self.times[2, 1] += 1
        if self.times[2, 1] % 100 == 0:
            print('env step time %.3f' % (self.times[2, 0] / self.times[2, 1]))

        if self.event.metadata['lastActionSuccess']:
            self.process_frame()

    def draw_state(self):
        from utils import drawing
        scale = 8
        if self.board is None:
            locs = self.graph.points * scale
            self.board = np.zeros(((self.graph.yMax - self.graph.yMin) * scale, (self.graph.xMax - self.graph.xMin) * scale), dtype=np.uint8)
            locs -= np.array([self.graph.xMin, self.graph.yMin]) * scale
            for loc in locs:
                drawing.drawRect(self.board, [loc[0], loc[1], loc[0], loc[1]], scale / 2, 4)
            if type(self.end_point) == list:
                for end_point in self.end_point:
                    goal_loc = (np.array(end_point) * np.array([scale, scale, 90]) -
                            np.array([self.graph.xMin, self.graph.yMin, 0]) * scale).astype(int)
                    drawing.drawRect(self.board, [goal_loc[0], goal_loc[1], goal_loc[0], goal_loc[1]], scale / 2, 5)
            else:
                goal_loc = (np.array(self.end_point) * np.array([scale, scale, 90]) -
                        np.array([self.graph.xMin, self.graph.yMin, 0]) * scale).astype(int)
                goal_arrow = [goal_loc[0] + scale / 2 * (goal_loc[2] == 90) - scale / 2 * (goal_loc[2] == 270),
                              goal_loc[1] + scale / 2 * (goal_loc[2] == 0) - scale / 2 * (goal_loc[2] == 180)]
                drawing.drawRect(self.board, [goal_loc[0], goal_loc[1], goal_loc[0], goal_loc[1]], scale / 2, 5)
                drawing.drawRect(self.board, [goal_arrow[0], goal_arrow[1], goal_arrow[0], goal_arrow[1]], scale / 4, 6)

        self.board[np.logical_or(self.board == 2, self.board == 3)] = 4
        curr_point = np.array(self.pose[:3])
        curr_loc = (curr_point * np.array([scale, scale, 90]) -
                  np.array([self.graph.xMin, self.graph.yMin, 0]) * scale).astype(int)
        curr_arrow = [curr_loc[0] + scale / 2 * (curr_loc[2] == 90) - scale / 2 * (curr_loc[2] == 270),
                    curr_loc[1] + scale / 2 * (curr_loc[2] == 0) - scale / 2 * (curr_loc[2] == 180)]
        drawing.drawRect(self.board, [curr_loc[0], curr_loc[1], curr_loc[0], curr_loc[1]], scale / 2, 2)
        drawing.drawRect(self.board, [curr_arrow[0], curr_arrow[1], curr_arrow[0], curr_arrow[1]], scale / 4, 3)
        self.board[0, 0] = 6
        return np.flipud(self.board)


class QuestionGameState(GameState):
    def __init__(self, sess=None, depth_scope=None):
        super(QuestionGameState, self).__init__(sess, depth_scope)

        self.question_types = ['existence', 'counting', 'contains']
        self.datasets = []
        self.test_datasets = []
        for qq,question_type in enumerate(self.question_types):
            prefix = 'questions/'
            path = prefix + 'train/data' + '_' + question_type
            print('path', path)
            data_file = sorted(glob.glob(path + '/*.h5'), key=os.path.getmtime)
            if len(data_file) > 0 and qq in constants.USED_QUESTION_TYPES:
                dataset = h5py.File(data_file[-1])
                dataset_np = dataset['questions/question'][...]
                dataset = dataset_np
                sums = np.sum(np.abs(dataset), axis=1)
                self.datasets.append(dataset[sums > 0])
                print('Type', question_type, 'num_questions', self.datasets[-1].shape)
            else:
                self.datasets.append([])

            # test data
            path = prefix + constants.TEST_SET + '/data' + '_' + question_type
            print('path', path)
            data_file = sorted(glob.glob(path + '/*.h5'), key=os.path.getmtime)
            if len(data_file) > 0 and qq in constants.USED_QUESTION_TYPES:
                dataset = h5py.File(data_file[-1])
                dataset_np = dataset['questions/question'][...]
                dataset.close()
                test_dataset = dataset_np
                sums = np.sum(np.abs(test_dataset), axis=1)
                self.test_datasets.append(test_dataset[sums > 0])
                print('Type', question_type, 'test num_questions', self.test_datasets[-1].shape)
            else:
                self.test_datasets.append([])

    def reset(self, seed=None, test_ind=None):
        self.board = None
        self.seen_object = False
        self.terminal = False
        self.opened_receptacles = set()
        self.closed_receptacles = set()
        self.seen_obj1 = set()
        self.seen_obj2 = set()
        self.visited_locations = set()
        self.can_end = False
        if seed is not None:
            self.local_random.seed(seed)

        # Do equal number of each question type in train.
        question_type_ind = self.local_random.sample(constants.USED_QUESTION_TYPES, 1)[0]

        # get random row
        if test_ind is not None:
            question_row, question_type_ind = test_ind
            question_type = self.question_types[question_type_ind]
            question_data = self.test_datasets[question_type_ind][question_row, :]
            test_ind = (question_row, question_type_ind)
        else:
            question_type = self.question_types[question_type_ind]
            question_row = self.local_random.randint(0, len(self.datasets[question_type_ind]) - 1)
            question_data = self.datasets[question_type_ind][question_row, :]

        container_ind = None

        if question_type_ind == 0 or question_type_ind == 1:
            scene_num, scene_seed, object_ind, answer = question_data
            self.question_target = object_ind
            if question_type_ind == 0:
                answer = bool(answer)

        elif question_type_ind == 2:
            scene_num, scene_seed, object_ind, container_ind, answer = question_data
            answer = bool(answer)
            self.question_target = (object_ind, container_ind)
        else:
            raise Exception('No question type found for type %d' % question_type_ind)

        self.scene_seed = scene_seed
        self.scene_num = scene_num

        self.object_target = object_ind
        self.parent_target = container_ind
        self.container_target = np.zeros(constants.NUM_CLASSES)
        self.direction_target = np.zeros(4)
        if container_ind is not None:
            self.container_target[container_ind] = 1

        self.question_type_ind = question_type_ind

        self.scene_name = 'FloorPlan%d' % scene_num
        grid_file = 'layouts/%s-layout.npy' % self.scene_name
        self.graph = graph_obj.Graph(grid_file, use_gt=False)
        self.xray_graph = graph_obj.Graph(grid_file, use_gt=True)

        self.bounds = [self.graph.xMin, self.graph.yMin,
            self.graph.xMax - self.graph.xMin + 1,
            self.graph.yMax - self.graph.yMin + 1]

        max_num_repeats = 1
        remove_prob = 0.5
        if question_type == 'existence':
            max_num_repeats = 10
            remove_prob = 0.25
        elif question_type == 'counting':
            max_num_repeats = constants.MAX_COUNTING_ANSWER + 1
            remove_prob = 0.5
        elif question_type == 'contains':
            max_num_repeats = 10
            remove_prob = 0.25
        self.event = game_util.reset(self.env, self.scene_name)
        self.agent_height = self.event.metadata['agent']['position']['y']
        self.camera_height = self.agent_height + constants.CAMERA_HEIGHT_OFFSET
        self.event = self.env.random_initialize(self.scene_seed, max_num_repeats=max_num_repeats, remove_prob=remove_prob)

        print('Type:', question_type, 'Row: ', question_row, 'Scene', self.scene_name, 'seed', scene_seed)
        print('Question:', game_util.get_question_str(question_type_ind, object_ind, container_ind))
        if self.question_type_ind == 2:
            print('Answer:', constants.OBJECTS[object_ind], 'in', constants.OBJECTS[container_ind], 'is', answer)
        else:
            print('Answer:', constants.OBJECTS[object_ind], 'is', answer)
        self.answer = answer

        # Verify answer
        if self.question_type_ind == 0:
            objs = game_util.get_objects_of_type(constants.OBJECTS[object_ind], self.event.metadata)
            computed_answer = len(objs) > 0
            requires_interaction = True
            for obj in objs:
                parent = obj['parentReceptacle'].split('|')[0]
                if parent not in {'Fridge', 'Cabinet', 'Microwave'}:
                    requires_interaction = False
                    break
        elif self.question_type_ind == 1:
            objs = game_util.get_objects_of_type(constants.OBJECTS[object_ind], self.event.metadata)
            computed_answer = len(objs)
            requires_interaction = True
        elif self.question_type_ind == 2:
            objs = game_util.get_objects_of_type(constants.OBJECTS[object_ind], self.event.metadata)
            if len(objs) == 0:
                computed_answer = False
                requires_interaction = constants.OBJECTS[self.question_target[1]] in {'Fridge', 'Cabinet', 'Microwave'}
            else:
                obj = objs[0]
                computed_answer = False
                for obj in objs:
                    requires_interaction = True
                    parent = obj['parentReceptacle'].split('|')[0]
                    if parent in constants.OBJECT_CLASS_TO_ID:
                        parent_ind = constants.OBJECT_CLASS_TO_ID[parent]
                        computed_answer = parent_ind == self.question_target[1]
                        if computed_answer:
                            if parent not in {'Fridge', 'Cabinet', 'Microwave'}:
                                requires_interaction = False
                            break
                    else:
                        computed_answer = False

        self.requires_interaction = requires_interaction

        try:
            assert self.answer == computed_answer, 'Answer does not match scene metadata'
        except AssertionError:
            print('Type:', question_type, 'Row: ', question_row, 'Scene', self.scene_name, 'seed', scene_seed)
            print('Answer', computed_answer, 'does not match expected value', self.answer,', did randomization process change?')
            pdb.set_trace()
            self.answer = computed_answer

        if constants.NUM_CLASSES > 1:
            self.hidden_items = set()
            objects = self.event.metadata['objects']
            for obj in objects:
                if obj['receptacle'] and obj['openable'] and not obj['isopen']:
                    for inside_obj in obj['receptacleObjectIds']:
                        self.hidden_items.add(inside_obj)

            objects = self.event.metadata['objects']
            for obj in objects:
                if obj['objectType'] not in constants.OBJECT_CLASS_TO_ID:
                    continue
                obj_bounds = game_util.get_object_bounds(obj, self.bounds)
                self.xray_graph.memory[obj_bounds[1]:obj_bounds[3],
                    obj_bounds[0]:obj_bounds[2],
                    constants.OBJECT_CLASS_TO_ID[obj['objectType']] + 1] = 1

        start_point = self.local_random.randint(0, self.graph.points.shape[0] - 1)
        start_point = self.graph.points[start_point, :].copy()
        self.start_point = (start_point[0], start_point[1], self.local_random.randint(0, 3))

        action = {'action': 'TeleportFull',
                  'x': self.start_point[0] * constants.AGENT_STEP_SIZE,
                  'y': self.agent_height,
                  'z': self.start_point[1] * constants.AGENT_STEP_SIZE,
                  'rotateOnTeleport': True,
                  'rotation': self.start_point[2] * 90,
                  'horizon': 30,
                  }
        self.event = self.env.step(action)

        self.process_frame()
        self.reward = 0
        self.end_point = []

    def get_action(self, action_or_ind):
        teleport_failure = False
        should_fail = False
        if type(action_or_ind) == int:
            action = copy.deepcopy(self.action_util.actions[action_or_ind])
        else:
            action = action_or_ind

        if action['action'] == 'Teleport':
            point_dists = np.sum(np.abs(self.graph.points - np.array([action['x'], action['z']])), axis=1)
            dist_min = np.argmin(point_dists)
            if point_dists[dist_min] < 0.0001 or constants.USE_NAVIGATION_AGENT:
                point_x = action['x']
                point_z = action['z']
            else:
                point_x = self.graph.points[dist_min][0]
                point_z = self.graph.points[dist_min][1]
                teleport_failure = True

            action = {
                'action': 'Teleport',
                'x': point_x * constants.AGENT_STEP_SIZE,
                'y': self.agent_height,
                'z': point_z * constants.AGENT_STEP_SIZE,
                'rotateOnTeleport': True,
                'rotation': action['rotation'],
                }

        elif action['action'] == 'OpenObject' or action['action'] == 'CloseObject':
            openable = [obj for obj in self.event.metadata['objects']
                    if (obj['visible'] and obj['openable'] and
                        (obj['isopen'] == (action['action'] == 'CloseObject')) and
                        obj['objectId'] in self.event.instance_detections2D)]
            if len(openable) > 0:
                boxes = np.array([self.event.instance_detections2D[obj['objectId']] for obj in openable])
                boxes_xywh = bb_util.xyxy_to_xywh(boxes.T).T
                mids = boxes_xywh[:, :2]
                dists = np.sqrt(np.sum(np.square(
                        (mids - np.array([constants.SCREEN_WIDTH / 2, constants.SCREEN_HEIGHT / 2]))), axis=1))
                obj_ind = np.argmin(dists)
                action['objectId'] = openable[obj_ind]['objectId']
            else:
                should_fail = True

        return action, teleport_failure, should_fail

    def step(self, action_or_ind):
        self.reward = -0.01
        action, teleport_failure, should_fail = self.get_action(action_or_ind)

        t_start = time.time()
        if should_fail or teleport_failure:
            self.event.metadata['lastActionSuccess'] = False
        else:
            if action['action'] != 'Teleport' or not constants.USE_NAVIGATION_AGENT:
                self.event = self.env.step(action)
            else:
                # Action is teleport and I should do low level navigation.
                pass

        new_pose = game_util.get_pose(self.event)
        point_dists = np.sum(np.abs(self.graph.points - np.array(new_pose)[:2]), axis=1)
        if np.min(point_dists) > 0.0001:
            print('Point teleport failure')
            closest_point = self.graph.points[np.argmin(point_dists)]
            self.event = self.env.step({
                'action': 'Teleport',
                'x': closest_point[0] * constants.AGENT_STEP_SIZE,
                'y': self.agent_height,
                'z': closest_point[1] * constants.AGENT_STEP_SIZE,
                'rotateOnTeleport': True,
                'rotation': self.pose[2] * 90,
            })
        else:
            closest_point = np.argmin(point_dists)
            if closest_point not in self.visited_locations:
                self.visited_locations.add(closest_point)

        self.times[2, 0] += time.time() - t_start
        self.times[2, 1] += 1
        if self.times[2, 1] % 100 == 0:
            print('env step time %.3f' % (self.times[2, 0] / self.times[2, 1]))

        if self.event.metadata['lastActionSuccess']:
            self.process_frame()

            if action['action'] == 'OpenObject':
                if self.question_type_ind == 2 and action['objectId'].split('|')[0] != self.question_target[1]:
                    self.reward -= 1.0
                elif action['objectId'] not in self.opened_receptacles:
                    if self.question_type_ind == 2 and action['objectId'].split('|')[0] == self.question_target[1]:
                        self.reward += 5.0
                    else:
                        self.reward += 0.1
                self.opened_receptacles.add(action['objectId'])

            elif action['action'] == 'CloseObject' and self.question_type_ind != 2:
                if action['objectId'] not in self.closed_receptacles:
                    self.reward += 0.1
                self.closed_receptacles.add(action['objectId'])

            # Update seen objects related to question
            objs = game_util.get_objects_of_type(constants.OBJECTS[self.object_target], self.event.metadata)
            objs = [obj for obj in objs if (obj['objectId'] in self.event.instance_detections2D and
                        game_util.check_object_size(self.event.instance_detections2D[obj['objectId']]))]
            for obj in objs:
                self.seen_obj1.add(obj['objectId'])
            if self.question_type_ind in {2, 3}:
                objs = game_util.get_objects_of_type(constants.OBJECTS[self.question_target[1]], self.event.metadata)
                objs = [obj for obj in objs if (obj['objectId'] in self.event.instance_detections2D and
                            game_util.check_object_size(self.event.instance_detections2D[obj['objectId']]))]
                for obj in objs:
                    self.seen_obj2.add(obj['objectId'])
            if not self.can_end:
                if self.question_type_ind == 0:
                    self.can_end = len(self.seen_obj1) > 0
                elif self.question_type_ind == 1:
                    self.can_end = len(self.seen_obj1) == self.answer
                elif self.question_type_ind == 2:
                    objs = game_util.get_objects_of_type(constants.OBJECTS[self.question_target[1]], self.event.metadata)
                    if not self.answer:
                        if objs[0]['openable']:
                            if all([obj['objectId'] in self.opened_receptacles for obj in objs]):
                                self.can_end = True
                        else:
                            if all([obj['objectId'] in self.seen_obj2 for obj in objs]):
                                self.can_end = True
                    else:
                        objs = [obj for obj in objs if (obj['objectId'] in self.event.instance_detections2D and
                                    game_util.check_object_size(self.event.instance_detections2D[obj['objectId']]))]
                        for obj in objs:
                            for contained_obj in obj['pivotSimObjs']:
                                if contained_obj['objectId'] in self.seen_obj1:
                                    self.can_end = True

        else:
            self.reward -= 0.05

