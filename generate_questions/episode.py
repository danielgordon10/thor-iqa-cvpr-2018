"""A wrapper for engaging with the THOR environment."""
import random

import numpy as np

from utils import game_util


class Episode(object):
    """Manages an episode in the THOR env."""

    def __init__(self):
        """Init function
           Inputs:
        """

        # Start the environment.
        self.env = self.start_env()
        self.event = None
        self.is_initialized = False

    def start_env(self):
        """Starts the environment."""
        env = game_util.create_env(quality='Very Low')
        return env

    def stop_env(self):
        """Stops the env."""
        self.env.stop()

    def get_objects(self):
        return self.event.metadata['objects']

    def initialize_scene(self, scene_name):
        self.scene_name = scene_name
        self.get_env_info()

    def get_env_info(self):
        """Get env specific information."""
        event = game_util.reset(self.env, self.scene_name,
                                render_depth_image=False,
                                render_class_image=False,
                                render_object_image=True)
        self.object_id_to_object_class = {
            obj['objectId']: obj['objectType'] for obj in event.metadata['objects']
        }

        self.pickable_object_classes = sorted(list(set([
            obj['objectType'] for obj in event.metadata['objects'] if obj['pickupable'] and obj['objectType'] != 'MiscObject'
        ])))
        print('# Pickable object_classes:',
              len(self.pickable_object_classes))

        # Find all receptacles.
        self.receptacles = sorted([
            obj['objectId'] for obj in event.metadata['objects']
            if obj['receptacle']
        ])
        print('# Receptacles:', len(self.receptacles))

        # Find all receptacle classes.
        self.receptacle_classes = list(set([item.split(
            '|')[0] for item in self.receptacles]))
        print('# Receptacle classes:', len(
            self.receptacle_classes))

        # Find all openable receptacles.
        self.openable_receptacles = sorted([
            obj['objectId'] for obj in event.metadata['objects']
            if obj['receptacle'] and obj['openable']
        ])
        print('# Openable Receptacles:', len(self.openable_receptacles))

        # Find all openable receptacle classes.
        self.openable_receptacle_classes = list(set([item.split(
            '|')[0] for item in self.openable_receptacles]))
        print('# Openable object_classes:', len(
            self.openable_receptacle_classes))
        self.agent_height = event.metadata['agent']['position']['y']

    def initialize_episode(self, scene_seed=None, agent_seed=None, max_num_repeats=10, remove_prob=0.25):
        """Initializes environment with given scene and random seed."""
        # Reset the scene with some random seed.
        if scene_seed is None:
            scene_seed = random.randint(0, 999999999)
        if agent_seed is None:
            agent_seed = random.randint(0, 999999999)
        self.event = game_util.reset(self.env, self.scene_name,
                                render_depth_image=False,
                                render_class_image=False,
                                render_object_image=True)
        self.event = self.env.random_initialize(random_seed=scene_seed, max_num_repeats=max_num_repeats, remove_prob=remove_prob)
        self.agent_height = self.event.metadata['agent']['position']['y']

        self.is_initialized = True

        return scene_seed, agent_seed

    def step(self, action_to_take, intercept=True, choose_object='closest', raise_for_failure=False):
        """Take required step and return reward, terminal, success flags.

           intercept: Whether to convert complex actions to basic ones.
           choose_object: When an object interaction command is given, how to
               select the object instance to interact with?
        """
        assert self.is_initialized, "Env not initialized."
        self.event, actual_action = self.env.step(
            action_to_take, intercept, choose_object)
        if raise_for_failure:
            assert self.event.metadata['lastActionSuccess']

    def get_agent_location(self):
        """Gets agent's location."""
        location = np.array([
            self.event.metadata['agent']['position']['x'],
            self.event.metadata['agent']['position']['y'],
            self.event.metadata['agent']['position']['z'],
            self.event.metadata['agent']['rotation']['y'],
            self.event.metadata['agent']['cameraHorizon']
        ])
        return location
