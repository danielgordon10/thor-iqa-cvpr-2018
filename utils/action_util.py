import pdb
import copy
import json
import numpy as np

from utils import game_util

import constants

class ActionUtil(object):
    def __init__(self):
        self.actions = [
                {'action' : 'MoveAhead', 'moveMagnitude' : constants.AGENT_STEP_SIZE},
                {'action' : 'RotateLeft'},
                {'action' : 'RotateRight'},
                #{'action' : 'LookUp'},
                #{'action' : 'LookDown'},
                ]
        self.action_to_ind = {frozenset(action.items()) : ii for ii,action in enumerate(self.actions)}

        self.reverse_actions = {
            'MoveAhead' : 'MoveBack',
            'MoveBack' : 'MoveAhead',
            'MoveLeft' : 'MoveRight',
            'MoveRight' : 'MoveLeft',
            'RotateLeft' : 'RotateRight',
            'RotateRight' : 'RotateLeft',
            'LookUp' : 'LookDown',
            'LookDown' : 'LookUp',
            'PickupObject' : 'PutObject',
            'PutObject' : 'PickupObject',
            'OpenObject' : 'CloseObject',
            'CloseObject' : 'OpenObject'
            }

        self.num_actions = len(self.actions)

    def action_dict_to_ind(self, action):
        return self.action_to_ind[frozenset(action.items())]


