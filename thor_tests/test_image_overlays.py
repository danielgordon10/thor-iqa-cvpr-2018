import os
import numpy as np
import scipy.misc
from utils import game_util

def assert_image(expected_image, actual_image, raise_if_fail=True):
    try:
        assert(np.all(expected_image == actual_image))
    except AssertionError as e:
        print('Failed image comparison')
        import cv2
        diff = (expected_image.astype(np.int32) - actual_image.astype(np.int32)).astype(np.float32)
        diff -= diff.min()
        diff *= 255 / diff.max()
        diff = diff.astype(np.uint8)
        if len(expected_image.shape) == 3:
            cv2.imshow('expected', expected_image[:,:,::-1])
            cv2.imshow('actual', actual_image[:,:,::-1])
            cv2.imshow('diff', diff[:,:,::-1])
        else:
            cv2.imshow('expected', expected_image)
            cv2.imshow('actual', actual_image)
            cv2.imshow('diff', diff)
        cv2.waitKey(0)
        if raise_if_fail:
            raise e


def test_depth_and_ids_images(env):
    def compare_images_for_scene(scene_num, coordinates):
        event = env.reset('FloorPlan%d' % scene_num)
        env.step(dict(
            action='Initialize',
            gridSize=0.25,
            cameraY=0.75,
            qualitySetting='MediumCloseFitShadows',
            renderImage=True, renderDepthImage=True, renderClassImage=True, renderObjectImage=True))
        event = env.random_initialize(random_seed=0)
        event = env.step(
                {
                    'action' : 'TeleportFull',
                    'x' : coordinates[0],
                    'y': 1.5,
                    'z': coordinates[1],
                    'rotaton': 0,
                    'horizon' : 0
                })
        event = env.step({'action': 'RotateLeft'})
        event = env.step({'action': 'MoveAhead'})

        quantized_depth = (event.depth_frame / 5000 * 255).astype(np.uint8)
        compare_image = scipy.misc.imread(os.path.join('thor_tests', 'test_images', 'test_image_depth_%d.png' % scene_num))
        assert_image(compare_image, quantized_depth)

        ids_image = event.class_segmentation_frame
        compare_image = scipy.misc.imread(os.path.join('thor_tests', 'test_images', 'test_image_ids_%d.png' % scene_num))
        assert_image(compare_image, ids_image)

    compare_images_for_scene(1, (1.5, -1))
    compare_images_for_scene(2, (1, 1))
    compare_images_for_scene(3, (1, 0))


def run_tests(env=None):
    create_env = (env is None)
    if create_env:
        env = game_util.create_env()
    test_depth_and_ids_images(env)
    print('All test_image_overlays tests passed!')


if __name__ == '__main__':
    run_tests()
