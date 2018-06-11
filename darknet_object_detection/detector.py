import glob
import numpy as np
import scipy.misc
import os
import time
import constants

import threading

from utils import bb_util
from utils import drawing
from utils import py_util

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
WEIGHT_PATH = os.path.join(DIR_PATH, 'yolo_weights/')


class ObjectDetector(object):
    def __init__(self, detector_num=0):
        import darknet as dn
        dn.set_gpu(int(constants.DARKNET_GPU))
        self.detector_num = detector_num
        self.net = dn.load_net(py_util.encode(WEIGHT_PATH + 'yolov3-thor.cfg'),
                               py_util.encode(WEIGHT_PATH + 'yolov3-thor_final.weights'), 0)
        self.meta = dn.load_meta(py_util.encode(WEIGHT_PATH + 'thor.data'))

        self.count = 0

    def detect(self, image, confidence_threshold=constants.DETECTION_THRESHOLD):
        import darknet as dn
        self.count += 1

        start = time.time()
        results = dn.detect_numpy(self.net, self.meta, image, thresh=confidence_threshold)

        if len(results) > 0:
            classes, scores, boxes = zip(*results)
        else:
            classes = []
            scores = []
            boxes = np.zeros((0, 4))
        boxes = np.array(boxes)
        scores = np.array(scores)
        classes = np.array([py_util.decode(cls) for cls in classes])
        inds = np.where(np.logical_and(scores > confidence_threshold,
                                       np.min(boxes[:, [2, 3]], axis=1) > .01 * image.shape[0]))[0]
        used_inds = []
        for ind in inds:
            if classes[ind] in constants.OBJECTS_SET:
                used_inds.append(ind)
        inds = np.array(used_inds)
        if len(inds) > 0:
            classes = np.array(classes[inds])
            boxes = boxes[inds]
            if len(boxes) > 0:
                boxes = bb_util.xywh_to_xyxy(boxes.T).T
            boxes *= np.array([constants.SCREEN_HEIGHT * 1.0 / image.shape[1],
                               constants.SCREEN_WIDTH * 1.0 / image.shape[0]])[[0, 1, 0, 1]]
            boxes = np.clip(np.round(boxes), 0, np.array([constants.SCREEN_WIDTH,
                                                          constants.SCREEN_HEIGHT])[[0, 1, 0, 1]]).astype(np.int32)
            scores = scores[inds]
        else:
            boxes = np.zeros((0, 4))
            classes = np.zeros(0)
            scores = np.zeros(0)
        return boxes, scores, classes

def visualize_detections(image, boxes, classes, scores):
    out_image = image.copy()
    if len(boxes) > 0:
        boxes = (boxes / np.array([constants.SCREEN_HEIGHT * 1.0 / image.shape[1],
                constants.SCREEN_WIDTH * 1.0 / image.shape[0]])[[0, 1, 0, 1]]).astype(np.int32)
    for ii,box in enumerate(boxes):
        drawing.draw_detection_box(out_image, box, classes[ii], confidence=scores[ii], width=2)
    return out_image


singleton_detector = None
detectors = []
def setup_detectors(num_detectors=1):
    global detectors
    for dd in range(num_detectors):
        detectors.append(ObjectDetector(dd))

detector_ind = 0
detector_lock = threading.Lock()
def get_detector():
    global detectors, detector_ind
    detector_lock.acquire()
    detector = detectors[detector_ind % len(detectors)]
    detector_ind += 1
    detector_lock.release()
    return detector


if __name__ == '__main__':
    # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
    PATH_TO_TEST_IMAGES_DIR = DIR_PATH + '/test_images'
    TEST_IMAGE_PATHS = sorted(glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, '*.jpg')))

    if not os.path.exists(DIR_PATH + '/test_images/output'):
        os.mkdir(DIR_PATH + '/test_images/output')

    setup_detectors()
    detector = get_detector()

    t_start = time.time()
    import cv2
    for image_path in TEST_IMAGE_PATHS:
        print('image', image_path)
        image = scipy.misc.imread(image_path)
        (boxes, scores, classes) = detector.detect(image)
        # Visualization of the results of a detection.
        image = visualize_detections(image, boxes, classes, scores)
        scipy.misc.imsave(DIR_PATH + '/test_images/output/' + os.path.basename(image_path), image)
    total_time = time.time() - t_start
    print('total time %.3f' % total_time)
    print('per image time %.3f' % (total_time / len(TEST_IMAGE_PATHS)))
