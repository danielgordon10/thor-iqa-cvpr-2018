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
from generate_questions.questions import CountQuestion

import constants

all_object_classes = constants.QUESTION_OBJECT_CLASS_LIST

DEBUG = False
if DEBUG:
    PARALLEL_SIZE = 1
else:
    PARALLEL_SIZE = 8


def main(dataset_type):
    if dataset_type == 'val/unseen_scenes':
        num_questions_per_scene = round(100.0 / PARALLEL_SIZE)
        scene_numbers = constants.TEST_SCENE_NUMBERS
        num_samples_per_scene = 8
    elif dataset_type == 'val/seen_scenes':
        num_questions_per_scene = round(8.0 / PARALLEL_SIZE)
        scene_numbers = constants.TRAIN_SCENE_NUMBERS
        num_samples_per_scene = 4
    elif dataset_type == 'train':
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
        if not os.path.exists(prefix + dataset_type + '/data_counting'):
            os.makedirs(prefix + dataset_type + '/data_counting')

        h5 = h5py.File(prefix + dataset_type + '/data_counting/Counting_Questions_' + time_str + '.h5', 'w')
        h5.create_dataset('questions/question', (num_record, 4), dtype=np.int32)
        print('Generating %d counting questions' % num_record)

        # Generate counting questions
        data_ind = 0
        episode = Episode()
        scene_number = random.randint(0, len(scene_numbers) - 1)
        while data_ind < num_record:
            k = 0

            scene_number += 1
            scene_num = scene_numbers[scene_number % len(scene_numbers)]

            scene_name = 'FloorPlan%d' % scene_num
            episode.initialize_scene(scene_name)
            num_tries = 0
            while num_tries < num_samples_per_scene:
                # randomly pick a pickable object in the scene
                object_class = random.choice(all_object_classes)
                question = CountQuestion(object_class)  # randomly generate a general counting question
                generated = [None] * (constants.MAX_COUNTING_ANSWER + 1)
                generated_counts = set()

                num_tries += 1

                grid_file = 'layouts/%s-layout.npy' % scene_name
                xray_graph = graph_obj.Graph(grid_file, use_gt=True, construct_graph=False)
                scene_bounds = [xray_graph.xMin, xray_graph.yMin,
                    xray_graph.xMax - xray_graph.xMin + 1,
                    xray_graph.yMax - xray_graph.yMin + 1]

                for i in range(100):
                    if DEBUG:
                        print('starting try ', i)
                    scene_seed = random.randint(0, 999999999)
                    episode.initialize_episode(scene_seed=scene_seed, max_num_repeats=constants.MAX_COUNTING_ANSWER + 1, remove_prob=0.5)
                    answer = question.get_answer(episode)
                    object_target = constants.OBJECT_CLASS_TO_ID[object_class]

                    if answer > 0 and answer not in generated_counts:
                        if DEBUG:
                            print('target', str(question), object_target, answer)
                        event = episode.event

                        # Make sure findable
                        try:
                            objs = {obj['objectId']: obj for obj in event.metadata['objects'] if obj['objectType'] == object_class}
                            xray_graph.memory[:, :, 1:] = 0
                            for obj in objs.values():
                                obj_point = game_util.get_object_point(obj, scene_bounds)
                                xray_graph.memory[obj_point[1], obj_point[0],
                                                  object_target + 1] = 1
                            start_graph = xray_graph.memory.copy()

                            graph_points = xray_graph.points.copy()
                            graph_points = graph_points[np.random.permutation(graph_points.shape[0]), :]
                            num_checked_points = 0
                            point_ind = 0

                            # Initial check to make sure all objects are visible on the grid.
                            while point_ind < len(graph_points):
                                start_point = graph_points[point_ind]
                                headings = np.random.permutation(4)
                                for heading in headings:
                                    start_point = (start_point[0], start_point[1], heading)
                                    patch = xray_graph.get_graph_patch(start_point)[0][:, :, object_target + 1]
                                    if patch.max() > 0:
                                        point_ind = 0
                                        xray_graph.update_graph((np.zeros((constants.STEPS_AHEAD, constants.STEPS_AHEAD, 1)), 0), start_point, [object_target + 1])
                                point_ind += 1
                            if np.max(xray_graph.memory[:, :, object_target + 1]) > 0:
                                if DEBUG:
                                    print('some points could not be reached')
                                answer = None
                                raise AssertionError

                            xray_graph.memory = start_graph
                            point_ind = 0
                            seen_objs = set()
                            while point_ind < len(graph_points):
                                start_point = graph_points[point_ind]
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
                                        if num_checked_points > 20:
                                            if DEBUG:
                                                print('timeout')
                                            answer = None
                                            raise AssertionError
                                        changed = False

                                        for jj in range(4):
                                            open_success = True
                                            opened_objects = set()
                                            parents = [game_util.get_object(obj['parentReceptacle'], event.metadata)
                                                       for obj in objs.values()]
                                            openable_parents = [parent for parent in parents
                                                                if parent['visible'] and parent['openable'] and not parent['isopen']]
                                            while open_success:
                                                obj_list = list(objs.values())
                                                for obj in obj_list:
                                                    if obj['objectId'] in event.instance_detections2D:
                                                        if game_util.check_object_size(event.instance_detections2D[obj['objectId']]):
                                                            seen_objs.add(obj['objectId'])
                                                            if DEBUG:
                                                                print('seen', seen_objs)
                                                            del objs[obj['objectId']]
                                                            changed = True
                                                            num_checked_points = 0
                                                            if len(seen_objs) == answer:
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
                                        if changed:
                                            point_ind = 0
                                            num_checked_points = 0
                                            xray_graph.memory[:, :, object_target + 1] = 0
                                            for obj in objs.values():
                                                obj_point = game_util.get_object_point(obj, scene_bounds)
                                                xray_graph.memory[obj_point[1], obj_point[0],
                                                                  object_target + 1] = 1
                                point_ind += 1
                            if DEBUG:
                                print('ran out of points')
                            answer = None
                        except AssertionError:
                            if answer is not None:
                                if DEBUG:
                                    print('success')
                            pass

                    print(str(question), object_target, answer)

                    if answer is not None and answer < len(generated) and answer not in generated_counts:
                        generated[answer] = [scene_num, scene_seed, constants.OBJECT_CLASS_TO_ID[object_class], answer]
                        generated_counts.add(answer)
                        print('\tcounts', sorted(list(generated_counts)))

                    if len(generated_counts) == len(generated):
                        for q in generated:
                            if data_ind >= h5['questions/question'].shape[0]:
                                num_tries = 2**32
                                break
                            h5['questions/question'][data_ind, :] = np.array(q)
                            data_ind += 1
                            k += 1
                        h5.flush()
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
            procs.append(proc)
            proc.start()
            time.sleep(1)
        for proc in procs:
            proc.join()


if __name__ == '__main__':
    main('train')
    main('val/unseen_scenes')
    main('val/seen_scenes')
