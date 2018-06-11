import pdb
import cv2
import numpy as np
import random
import os

from utils import game_util
from utils import drawing
import constants


class PersonGameState(object):
    def __init__(self):
        self.env = game_util.create_env()
        self.env.step({'action': 'Initialize', 'gridSize': constants.AGENT_STEP_SIZE})
        self.local_random = random.Random()

        self.question_types = ['existence', 'counting', 'contains']
        self.datasets = []
        self.test_datasets = []
        self.num_steps = 0
        self.num_failed_steps = 0
        for (qq, question_type) in enumerate(self.question_types):
            data_file = os.path.join('questions', 'train', 'data' + '_' + question_type, 'combined.csv')
            if qq in constants.USED_QUESTION_TYPES:
                dataset = [line.strip().split(',') for line in open(data_file)][1:]
                self.datasets.append(dataset)
                print('Type', question_type, 'num_questions', len(self.datasets[-1]))
            else:
                self.datasets.append([])

            # test data
            data_file = os.path.join('questions', constants.TEST_SET, 'data' + '_' + question_type, 'combined.csv')
            if qq in constants.USED_QUESTION_TYPES:
                dataset = [line.strip().split(',') for line in open(data_file)][1:]
                self.test_datasets.append(dataset)
                print('Type', question_type, 'num_questions', len(self.test_datasets[-1]))
            else:
                self.test_datasets.append([])

    def reset(self, seed=None, test_ind=None):
        if seed is not None:
            self.local_random.seed(seed)

        question_row, question_type_ind = test_ind
        question_type = self.question_types[question_type_ind]
        question_data = self.test_datasets[question_type_ind][question_row % len(self.test_datasets[question_type_ind])]
        scene_num, scene_seed, question_str, answer = question_data[1:5]

        self.scene_seed = int(scene_seed)
        self.scene_num = int(scene_num)
        self.question_str = question_str

        self.question_type_ind = question_type_ind

        self.scene_name = 'FloorPlan%d' % self.scene_num
        grid_file = 'layouts/%s-layout.npy' % self.scene_name
        self.points = (np.load(grid_file) * 1.0 / constants.AGENT_STEP_SIZE).astype(int)

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
        self.event = game_util.reset(self.env, self.scene_name,
                                     render_image=True,
                                     render_depth_image=True,
                                     render_class_image=True,
                                     render_object_image=True)
        self.agent_height = self.event.metadata['agent']['position']['y']
        self.event = self.env.random_initialize(self.scene_seed, max_num_repeats=max_num_repeats, remove_prob=remove_prob)

        print('Question: %s' % self.question_str)

        if answer == 'True':
            self.answer = True
        elif answer == 'False':
            self.answer = True
        else:
            self.answer = int(answer)

        start_point = self.local_random.randint(0, self.points.shape[0] - 1)
        start_point = self.points[start_point, :].copy()
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

    def step(self, action_key):
        action = None
        if action_key == 'w':
            action = {'action': 'MoveAhead'}
        elif action_key == 'a':
            action = {'action': 'RotateLeft'}
        elif action_key == 's':
            action = {'action': 'RotateRight'}
        elif action_key == 'o':
            action = {'action': 'OpenObject'}
        elif action_key == 'c':
            action = {'action': 'CloseObject'}
        elif action_key == '+':
            action = {'action': 'LookUp'}
        elif action_key == '-':
            action = {'action': 'LookDown'}
        elif action_key == 'answer':
            pass
        elif action_key == 'q':
            quit()
        elif action_key == 'dd':
            import pdb
            pdb.set_trace()
            print('debug entered')
        else:
            return
        self.num_steps += 1
        if action is not None:
            if action['action'] in {'OpenObject', 'CloseObject'}:
                action = game_util.set_open_close_object(action, self.event)
            self.event = self.env.step(action)
            if not self.event.metadata['lastActionSuccess']:
                self.num_failed_steps += 1
        self.process_frame()

    def process_frame(self):
        self.pose = self.event.pose

        self.s_t = self.event.frame
        self.detection_image = self.s_t.copy()
        self.s_t_depth = self.event.frame_depth

        boxes = []
        scores = []
        class_names = []
        for k,v in self.event.class_detections2D.items():
            if k in constants.OBJECTS_SET:
                boxes.extend(v)
                scores.extend([1] * len(v))
                class_names.extend([k] * len(v))
        detected_objects = [game_util.get_object(obj_id, self.event.metadata) for obj_id in self.event.instance_detections2D.keys()]
        detected_objects = [obj for obj in detected_objects if obj is not None]
        boxes = np.array([self.event.instance_detections2D[obj['objectId']] for obj in detected_objects])
        class_names = np.array([obj['objectType'] for obj in detected_objects])
        scores = np.ones(len(boxes))
        self.detection_image = drawing.visualize_detections(
            self.event.frame, boxes, class_names, scores)
        print(self.question_str)


if __name__ == '__main__':
    state = PersonGameState()
    random.seed(0)
    for question_type in constants.USED_QUESTION_TYPES:
        print('Starting question type', question_type)
        num_correct = 0
        num_total = 0
        questions = []
        for test_ep in range(10):
            questions.append((test_ep, (random.randint(0, 2**31), question_type)))
        random.shuffle(questions)
        for (qq, question) in enumerate(questions):
            num_total += 1
            action_key = ''
            state.reset(*question)
            while action_key != 'answer':
                if constants.DEBUG:
                    images = [
                            state.s_t,
                            state.detection_image,
                            state.s_t_depth,
                            state.event.class_segmentation_frame,
                            state.event.instance_segmentation_frame]
                    titles = ['state', 'detections', 'depth', 'class segmentation', 'instance segmentation']
                    image = drawing.subplot(images, 2, 3, constants.SCREEN_WIDTH, constants.SCREEN_HEIGHT, 5, titles)

                    cv2.imshow('image', image[:, :, ::-1])
                    cv2.waitKey(10)
                print('w: MoveAhead\na: RotateLeft\ns: RotateRight\no: OpenObject\nc: CloseObject\n+: LookUp\n-: LookDown\nanswer: Open answer dialog. type {true, false, yes, no}\nq: quit\ndd: enter debug')
                new_action_key = input(">> ")
                if new_action_key != '':
                    action_key = new_action_key
                state.step(action_key)
            answer = None
            while answer is None:
                answer = input("answer: ").lower()
                if answer in {'true', 'false', 'yes', 'no'}:
                    if ((answer in {'true', 'yes'} and state.answer) or
                            (answer in {'false', 'no'} and not state.answer)):
                        print('Correct')
                        num_correct += 1
                else:
                    try:
                        answer = int(answer)
                        if answer == state.answer:
                            print('Correct')
                            num_correct += 1
                    except ValueError as ve:
                        answer = None
            print('Num questions', num_total)
            print('Correct percent: %.2f%%' % (num_correct * 100.0 / num_total))
            print('Total moves:', state.num_steps)
            print('Average moves:', (state.num_steps / (qq + 1)))
            print('Invalid moves percent: %.2f%%' % (state.num_failed_steps * 100.0 / state.num_steps))
            state.num_steps = 0
            state.num_failed_steps = 0
