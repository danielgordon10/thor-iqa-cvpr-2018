import pdb
import os
import time
from generate_questions.episode import Episode
import random
import numpy as np
import h5py
import multiprocessing
from utils import game_util
from utils import py_util
from graph import graph_obj
from generate_questions.questions import ExistenceQuestion
import copy

import constants

all_object_classes = constants.QUESTION_OBJECT_CLASS_LIST
receptacles = copy.deepcopy(constants.RECEPTACLES)

# Remove the ones that don't have good gt object detectors
receptacles.remove('Sink')
receptacles.remove('TableTop')
receptacles.remove('StoveBurner')

DEBUG = False
if DEBUG:
    PARALLEL_SIZE = 1
else:
    PARALLEL_SIZE = 8


def main(dataset_type):
    if dataset_type == 'test':
        num_questions_per_scene = round(100.0 / PARALLEL_SIZE)
        scene_numbers = constants.TEST_SCENE_NUMBERS
        num_samples_per_scene = 8
    elif dataset_type == 'val/seen_scenes':
        num_questions_per_scene = round(8.0 / PARALLEL_SIZE)
        scene_numbers = constants.TRAIN_SCENE_NUMBERS
        num_samples_per_scene = 4
    elif dataset_type == 'val/unseen_scenes':
        num_questions_per_scene = round(1000.0 / PARALLEL_SIZE)
        scene_numbers = constants.TRAIN_SCENE_NUMBERS
        num_samples_per_scene = 16
    else:
        raise Exception('No test set found')
    num_record = int(num_samples_per_scene * np.ceil(num_questions_per_scene * 1.0 / num_samples_per_scene) * len(scene_numbers))

    assert(num_samples_per_scene % 4 == 0)

    def create_dump():
        time_str = py_util.get_time_str()
        prefix = 'questions/'
        if not os.path.exists(prefix + dataset_type + '/data_contains'):
            os.makedirs(prefix + dataset_type + '/data_contains')

        h5 = h5py.File(prefix + dataset_type + '/data_contains/Contains_Questions_' + time_str + '.h5', 'w')
        h5.create_dataset('questions/question', (num_record, 5), dtype=np.int32)
        print('Generating %d contains questions' % num_record)

        # Generate contains questions
        data_ind = 0
        episode = Episode()
        scene_number = -1
        while data_ind < num_record:
            k = 0

            scene_number += 1
            scene_num = scene_numbers[scene_number % len(scene_numbers)]

            scene_name = 'FloorPlan%d' % scene_num
            episode.initialize_scene(scene_name)
            num_tries = 0
            while k < num_samples_per_scene and num_tries < 10 * num_samples_per_scene:
                # randomly pick a pickable object in the scene
                object_class = random.choice(all_object_classes)
                generated_no, generated_yes = False, False
                question = None
                temp_data = []
                num_tries += 1

                grid_file = 'layouts/%s-layout.npy' % scene_name
                xray_graph = graph_obj.Graph(grid_file, use_gt=True, construct_graph=False)
                scene_bounds = [xray_graph.xMin, xray_graph.yMin,
                    xray_graph.xMax - xray_graph.xMin + 1,
                    xray_graph.yMax - xray_graph.yMin + 1]

                for i in range(20):  # try 20 times
                    scene_seed = random.randint(0, 999999999)
                    episode.initialize_episode(scene_seed=scene_seed)  # randomly initialize the scene
                    if not question:
                        question = ExistenceQuestion.get_true_contain_question(episode, all_object_classes, receptacles)
                        if DEBUG:
                            print(str(question))
                        if not question:
                            continue
                        object_class = question.object_class
                        parent_class = question.parent_object_class

                    answer = question.get_answer(episode)
                    object_target = constants.OBJECT_CLASS_TO_ID[object_class]

                    if answer and not generated_yes:
                        event = episode.event
                        xray_graph.memory[:, :, 1:] = 0
                        objs = {obj['objectId']: obj for obj in event.metadata['objects']
                                if obj['objectType'] == object_class and obj['parentReceptacle'].split('|')[0] == parent_class}
                        for obj in objs.values():
                            if obj['objectType'] != object_class:
                                continue
                            obj_point = game_util.get_object_point(obj, scene_bounds)
                            xray_graph.memory[obj_point[1], obj_point[0],
                                    constants.OBJECT_CLASS_TO_ID[obj['objectType']] + 1] = 1

                        # Make sure findable
                        try:
                            graph_points = xray_graph.points.copy()
                            graph_points = graph_points[np.random.permutation(graph_points.shape[0]), :]
                            num_checked_points = 0
                            for start_point in graph_points:
                                headings = np.random.permutation(4)
                                for heading in headings:
                                    start_point = (start_point[0], start_point[1], heading)
                                    patch = xray_graph.get_graph_patch(start_point)[0]
                                    if patch[:, :, object_target + 1].max() > 0:
                                        action = {'action': 'TeleportFull',
                                                  'x': start_point[0] * constants.AGENT_STEP_SIZE,
                                                  'y': episode.agent_height,
                                                  'z': start_point[1] * constants.AGENT_STEP_SIZE,
                                                  'rotateOnTeleport': True,
                                                  'rotation': start_point[2] * 90,
                                                  'horizon': -30,
                                                  }
                                        event = episode.env.step(action)
                                        num_checked_points += 1
                                        if num_checked_points > 1000:
                                            answer = None
                                            raise AssertionError
                                        for jj in range(4):
                                            open_success = True
                                            opened_objects = set()
                                            parents = [game_util.get_object(obj['parentReceptacle'], event.metadata)
                                                       for obj in objs.values()]
                                            openable_parents = [parent for parent in parents
                                                                if parent['visible'] and parent['openable'] and not parent['isopen']]
                                            while open_success:
                                                for obj in objs.values():
                                                    if obj['objectId'] in event.instance_detections2D:
                                                        if game_util.check_object_size(event.instance_detections2D[obj['objectId']]):
                                                            raise AssertionError
                                                if len(openable_parents) > 0:
                                                    action = {'action': 'OpenObject'}
                                                    game_util.set_open_close_object(action, event)
                                                    event = episode.env.step(action)
                                                    open_success = event.metadata['lastActionSuccess']
                                                    if open_success:
                                                        opened_objects.add(episode.env.last_action['objectId'])
                                                else:
                                                    open_success = False
                                            for opened in opened_objects:
                                                event = episode.env.step({
                                                    'action': 'CloseObject',
                                                    'objectId': opened,
                                                    'forceVisible': True})
                                                if not event.metadata['lastActionSuccess']:
                                                    answer = None
                                                    raise AssertionError
                                            if jj < 3:
                                                event = episode.env.step({'action': 'LookDown'})
                            answer = None
                        except AssertionError:
                            if answer is None and DEBUG:
                                print('failed to generate')

                    print(str(question), answer)

                    if answer == False and not generated_no:
                        generated_no = True
                        temp_data.append([scene_num, scene_seed, constants.OBJECT_CLASS_TO_ID[object_class], constants.OBJECT_CLASS_TO_ID[parent_class], answer])
                    elif answer == True and not generated_yes:
                        generated_yes = True
                        temp_data.append([scene_num, scene_seed, constants.OBJECT_CLASS_TO_ID[object_class], constants.OBJECT_CLASS_TO_ID[parent_class], answer])

                    if generated_no and generated_yes:
                        h5['questions/question'][data_ind, :] = np.array(temp_data[0])
                        h5['questions/question'][data_ind + 1, :] = np.array(temp_data[1])
                        h5.flush()
                        data_ind += 2
                        k += 2
                        break
                print("# generated samples: {}".format(data_ind))

        h5.close()
        episode.env.stop_unity()

    if DEBUG:
        create_dump()
    else:
        procs = []
        for ps in range(PARALLEL_SIZE):
            proc = multiprocessing.Process(target=create_dump)
            proc.start()
            procs.append(proc)
            time.sleep(1)
        for proc in procs:
            proc.join()


if __name__ == '__main__':
    main('train')
    main('val/unseen_scenes')
    main('val/seen_scenes')
