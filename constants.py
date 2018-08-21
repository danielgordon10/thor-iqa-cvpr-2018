# -*- coding: utf-8 -*-
import numpy as np
import os

# Parameters dictating which task should be run
# options are:
#   - navigation: train the navigation agent
#   - language_model: train the language model
#   - question_map_dump: generate data for semantic_map_pretraining
#   - semantic_map_pretraining: pretrain the answerer on gt semantic maps
#   - rl: use the rl controller
#   - end_to_end_baseline: run the A3C style agent
TASK = 'rl'
DEBUG = False
EVAL = False


########################################################################################################################
# Log file base directory info
LOG_PREFIX = 'logs'
CHECKPOINT_PREFIX = os.path.join(LOG_PREFIX, 'checkpoints/')
CHECKPOINT_DIR = CHECKPOINT_PREFIX[:]
LOG_FILE = os.path.join(LOG_PREFIX, 'train/')

if TASK in {'question_map_dump', 'semantic_map_pretraining'}:
    CHECKPOINT_DIR += 'rl'
    LOG_FILE += 'rl'
else:
    CHECKPOINT_DIR += TASK
    LOG_FILE += TASK

########################################################################################################################
# Ablation parameters
USE_NAVIGATION_AGENT = True
QUESTION_IN_ACTION = True
USE_POSSIBLE_PRIOR = True
OBJECT_DETECTION = True
GT_OBJECT_DETECTION = False

USED_QUESTION_TYPES = {0, 1, 2}

TEST_SET = 'test'  # options train_test -> unseen questions on seen rooms, test -> unseen rooms

PREDICT_DEPTH = not GT_OBJECT_DETECTION

########################################################################################################################
# Task specific hyperparameters
# Defaults
RENDER_DEPTH_IMAGE = False
RENDER_CLASS_IMAGE = False
RENDER_OBJECT_IMAGE = False

# Per task
if TASK == 'navigation':
    RL = False
    SUPERVISED = True
    USE_NAVIGATION_AGENT = True

    OBJECT_DETECTION = False
    GT_OBJECT_DETECTION = False
    PREDICT_DEPTH = False
    END_TO_END_BASELINE = False
    MAX_TIME_STEP = 5000
    MAX_EPISODE_LENGTH = 100
    NUM_EVAL_EPISODES = 100  # number of episodes for evaluation
    NUM_UNROLLS = 10
    BATCH_SIZE = 3 if DEBUG else 10
    REPLAY_BUFFER_SIZE = BATCH_SIZE * 10 if DEBUG else 128

    RENDER_DEPTH_IMAGE = False
    RENDER_CLASS_IMAGE = False
    RENDER_OBJECT_IMAGE = False

elif TASK == 'language_model':
    BATCH_SIZE = 1024
    MAX_TIME_STEP = 10000
    LEARNING_RATE = 1e-4
    RL = False

elif TASK == 'question_map_dump':
    RL = True
    END_TO_END_BASELINE = False
    RECORD_FEED_DICT = True
    MAX_TIME_STEP = int(1e9)
    SUPERVISED = True
    MAX_EPISODE_LENGTH = 4
    NUM_EVAL_EPISODES = 100  # number of episodes for evaluation
    GT_OBJECT_DETECTION = False
    OBJECT_DETECTION = False

    RENDER_DEPTH_IMAGE = False
    RENDER_CLASS_IMAGE = False
    RENDER_OBJECT_IMAGE = False

elif TASK == 'semantic_map_pretraining':
    MAX_TIME_STEP = 10000
    RL = True
    END_TO_END_BASELINE = False
    SUPERVISED = True
    BATCH_SIZE = 3 if DEBUG else 128

elif TASK == 'rl':
    RL = True
    SUPERVISED = False
    MAX_TIME_STEP = int(1e9)
    MAX_EPISODE_LENGTH = 1000
    END_TO_END_BASELINE = False
    NUM_UNROLLS = 12
    BATCH_SIZE = 3 if DEBUG else 8
    RECORD_FEED_DICT = False
    assert(OBJECT_DETECTION or GT_OBJECT_DETECTION), 'some sort of object detection must be used'
    assert(not OBJECT_DETECTION or not GT_OBJECT_DETECTION), 'only one object detection method should be used'

    RENDER_DEPTH_IMAGE = not PREDICT_DEPTH
    RENDER_CLASS_IMAGE = False
    RENDER_OBJECT_IMAGE = True

elif TASK == 'end_to_end_baseline':
    RL = True
    SUPERVISED = False
    RECORD_FEED_DICT = False
    END_TO_END_BASELINE = True
    MAX_TIME_STEP = int(1e9)
    MAX_EPISODE_LENGTH = 1000
    USE_NAVIGATION_AGENT = False
    NUM_UNROLLS = 32
    BATCH_SIZE = 3 if DEBUG else 16
    USE_OBJECT_DETECTION_AS_INPUT = True

    RENDER_DEPTH_IMAGE = False
    RENDER_CLASS_IMAGE = False
    RENDER_OBJECT_IMAGE = True

else:
    raise Exception('Unrecognized task type')


RENDER_IMAGE = True  # Pretty much always want to get the main camera image.

########################################################################################################################
# Run environment settings

if DEBUG:
    X_DISPLAY = '0.0'
    GPU_ID = '0'
    DARKNET_GPU = 0
    TRAIN = True
    DRAWING = True
    RUN_TEST = False
    if RUN_TEST:
        DEBUG = False
    PARALLEL_SIZE = 1
elif not EVAL:
    X_DISPLAY = '0.0'
    DARKNET_GPU = 0
    if RL:
        GPU_ID = '0'
        PARALLEL_SIZE = 2
    else:
        GPU_ID = '0'
        PARALLEL_SIZE = 8
    TRAIN = True
    DRAWING = False
    RUN_TEST = False
else:
    X_DISPLAY = '0.0'
    GPU_ID = '0'
    DARKNET_GPU = 1
    TRAIN = False
    DRAWING = False
    PARALLEL_SIZE = 8


########################################################################################################################
# Unity Hyperparameters
AGENT_STEP_SIZE = 0.25
BUILD_PATH = 'thor_builds/build_linux/thor_full.x86_64'
SCREEN_WIDTH = 300
SCREEN_HEIGHT = 300
CAMERA_HEIGHT_OFFSET = 0.75

########################################################################################################################
# Network hyperparameters
MEMORY_SIZE = 32
QA_MEMORY_SIZE = 256
GRU_SIZE = 1024
RL_GRU_SIZE = 64
CURIOSITY_FEATURE_SIZE = 64
TERMINAL_THRESH = 0.8
POSSIBLE_THRESH = 0.1
STEPS_AHEAD = 5
SCENE_PADDING = 5
NUM_ACTIONS = STEPS_AHEAD ** 2 + 7  # rot_left, rot_right, look_up, look_down, open, close, answer
TERMINAL_CHECK_PADDING = SCENE_PADDING - 1

########################################################################################################################

if RL:
    if OBJECT_DETECTION:
        CHECKPOINT_DIR += '_det'
        LOG_FILE += '_det'
    if END_TO_END_BASELINE:
        CHECKPOINT_DIR += '_e2e_baseline'
        LOG_FILE += '_e2e_baseline'
    else:
        if QUESTION_IN_ACTION:
            CHECKPOINT_DIR += '_q_in_a'
            LOG_FILE += '_q_in_a'

        if USE_NAVIGATION_AGENT:
            CHECKPOINT_DIR += '_nav'
            LOG_FILE += '_nav'

        if USE_POSSIBLE_PRIOR:
            CHECKPOINT_DIR += '_upp'
            LOG_FILE += '_upp'


########################################################################################################################
# A3C Hyperparameters
if RL:
    GAMMA = 0.99  # discount factor for rewards
    ENTROPY_BETA = 0.1
    RMSP_ALPHA = 0.99  # decay parameter for RMSProp
    RMSP_EPSILON = 0.1  # epsilon parameter for RMSProp
    GRAD_NORM_CLIP = 10.0  # gradient norm clipping
    LOCAL_T_MAX = 32

########################################################################################################################
# Object detection hyperparameters
DETECTION_THRESHOLD = 0.7
MIN_DETECTION_LEN = 10
if GT_OBJECT_DETECTION:
    MIN_DETECTION_LEN = 4


########################################################################################################################
# Parameters that should not be changed
SPATIAL_MAP_WIDTH = 28
SPATIAL_MAP_HEIGHT = 26
SPATIAL_MAP_WIDTH += 4 * SCENE_PADDING + 1
SPATIAL_MAP_HEIGHT += 4 * SCENE_PADDING + 1
FOV = 60
FOCAL_LENGTH = SCREEN_HEIGHT / (2 * np.tan(FOV / 2 * np.pi / 180))
MAX_DEPTH = 5000
PREDICT_DEPTH_SLOPE = 1.1885502493692868   # Computed multiplicative bias for depth prediction network
PREDICT_DEPTH_INTERCEPT = 228.55248115021686 # Computed bias for depth prediction network

TRAIN_SCENE_NUMBERS = list(range(6, 31))

TEST_SCENE_NUMBERS = list(range(1, 6))

SCENE_NUMBERS = TRAIN_SCENE_NUMBERS if not EVAL else TEST_SCENE_NUMBERS

OBJECTS = [
    'Background',
    'Spoon',
    'Potato',
    'Fork',
    'Plate',
    'Egg',
    'Tomato',
    'Bowl',
    'Lettuce',
    'Apple',
    'Knife',
    'Container',
    'Bread',
    'Mug',
    'Sink',
    'StoveBurner',
    'TableTop',
    'GarbageCan',
    'Microwave',
    'Fridge',
    'Cabinet',
    ]
OBJECTS_SINGULAR = [
    'background',
    'spoon',
    'potato',
    'fork',
    'plate',
    'egg',
    'tomato',
    'bowl',
    'head of lettuce',
    'apple',
    'knife',
    'container',
    'loaf of bread',
    'mug',
    'sink',
    'stove burner',
    'table top',
    'garbage can',
    'microwave',
    'fridge',
    'cabinet',
    ]

OBJECTS_PLURAL = [
    'background',
    'spoons',
    'potatoes',
    'forks',
    'plates',
    'eggs',
    'tomatoes',
    'bowls',
    'heads of lettuce',
    'apples',
    'knives',
    'containers',
    'loaves of bread',
    'mugs',
    'sinks',
    'stove burners',
    'table tops',
    'garbage cans',
    'microwaves',
    'fridges',
    'cabinets',
    ]

OBJECTS_SET = set(OBJECTS)
OBJECT_CLASS_TO_ID = {obj: ii for (ii, obj) in enumerate(OBJECTS)}

RECEPTACLES = {'Sink', 'StoveBurner', 'TableTop', 'GarbageCan', 'Microwave', 'Fridge', 'Cabinet'}

NUM_RECEPTACLES = len(RECEPTACLES)
NUM_OBJECTS = len(OBJECTS) - NUM_RECEPTACLES - 2
NUM_CLASSES = len(OBJECTS)

# For generating questions
QUESTION_OBJECT_CLASS_LIST = [
    'Spoon',
    'Potato',
    'Fork',
    'Plate',
    'Egg',
    'Tomato',
    'Bowl',
    'Lettuce',
    'Apple',
    'Knife',
    'Container',
    'Bread',
    'Mug',
    ]

QUESTION_OBJECT_CLASS_TO_ID = {obj: ii for (ii, obj) in enumerate(QUESTION_OBJECT_CLASS_LIST)}
# List of openable classes.
OPENABLE_CLASS_LIST = ['Fridge', 'Cabinet', 'Microwave']
OPENABLE_CLASS_SET = set(OPENABLE_CLASS_LIST)

# Question stuff
MAX_SENTENCE_LENGTH = 17
MAX_COUNTING_ANSWER = 3

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
