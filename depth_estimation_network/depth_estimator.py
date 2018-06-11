import pdb
import numpy as np
import cv2
import os
import tensorflow as tf
from utils import tf_util

import depth_estimation_network.models as models
from constants import SCREEN_WIDTH, SCREEN_HEIGHT, PREDICT_DEPTH_SLOPE, PREDICT_DEPTH_INTERCEPT
import threading

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


class DepthEstimator(object):
    def __init__(self, sess):
        self.sess = sess
        self.input = tf.placeholder(tf.float32, shape=(None, SCREEN_HEIGHT, SCREEN_WIDTH, 3))
        self.net = models.ResNet50UpProj({'data': self.input}, None, 1, False, False)
        self.output = self.net.get_output()
        print('Loading depth model')
        self.lock = threading.Lock()

    def load_weights(self):
        # Use to load from ckpt file
        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(DIR_PATH + '/depth_network_weights')
        tf_util.restore(self.sess, checkpoint.model_checkpoint_path)
        print('Depth model loaded')

    def get_depth(self, image):
        self.lock.acquire()
        image = image.squeeze()
        if len(image.shape) == 3:
            image = image[np.newaxis, ...]

        pred = self.sess.run(self.output, feed_dict={self.input: image})[:, :, :, 0] * 1000
        pred *= PREDICT_DEPTH_SLOPE
        pred += PREDICT_DEPTH_INTERCEPT
        self.lock.release()
        pred = cv2.resize(pred.transpose(1, 2, 0), (SCREEN_WIDTH, SCREEN_HEIGHT))
        if len(pred.shape) == 3:
            pred = pred.transpose(2, 0, 1)
        return pred


singleton_depth_estimator = None
def get_depth_estimator(sess=None):
    global singleton_depth_estimator
    if sess is None:
        sess = tf_util.Session()
    with tf.variable_scope('', reuse=(singleton_depth_estimator is not None)):
        singleton_depth_estimator = DepthEstimator(sess)
    return singleton_depth_estimator


if __name__ == '__main__':
    estimator = get_depth_estimator()
    estimator.load_weights()
    import glob
    import scipy.misc
    test_images = sorted(glob.glob(DIR_PATH + '/test_images/*.jpg'))
    gt_depth_images = sorted(glob.glob(DIR_PATH + '/test_images/gt_depth/*.jpg'))
    images = []
    for image_path in test_images:
        image = scipy.misc.imread(image_path)
        image = cv2.resize(image, (SCREEN_WIDTH, SCREEN_HEIGHT))
        images.append(image)
    images = np.array(images)
    depths = estimator.get_depth(images)
    for ii in range(images.shape[0]):
        image = images[ii,...]
        depth = depths[ii,...]
        depth[depth > 5000] = 5000
        depth /= 5000
        depth *= 255
        depth = depth.astype(np.uint8)
        gt_depth = cv2.resize(scipy.misc.imread(gt_depth_images[ii]), (SCREEN_WIDTH, SCREEN_HEIGHT))
        cv2.imshow('image', image[:, :, ::-1])
        cv2.imshow('depth', depth)
        cv2.imshow('gt_depth', gt_depth)
        cv2.waitKey(0)








