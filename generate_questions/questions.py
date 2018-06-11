import pdb
import random
import numpy as np
from utils import game_util

from constants import OBJECT_CLASS_TO_ID

class Question(object):
    def __init__(self):
        pass
    def get_answer(self, episode):
        raise NotImplementedError
    def __str__(self):
        raise NotImplementedError

class ExistenceQuestion(Question):
    def __init__(self, object_class, parent_object_class = None):
        super(ExistenceQuestion, self).__init__()
        #assert parent_object_class == None or object_class.name in parent_object_class.children

        self.object_class = object_class
        self.parent_object_class = parent_object_class
        if parent_object_class == 'TableTop':
            self.preposition='on'
        else:
            self.preposition = 'in'

    @staticmethod
    def get_true_contain_question(episode, things, receptacles):
        receptacles = list(receptacles)
        random.shuffle(receptacles)
        for parent in receptacles:
            parents = game_util.get_objects_of_type(parent, episode.event.metadata)
            random.shuffle(parents)
            for parent in parents:
                if len(parent['pivotSimObjs']) == 0:
                    continue
                obj = random.choice(parent['pivotSimObjs'])
                obj_type = obj['objectId'].split('|')[0]
                if obj_type in things:
                    return ExistenceQuestion(obj_type, parent['objectType'])
        return None

    def get_answer(self, episode):
        """ Get answer to the question given an episode """
        if self.parent_object_class is None:
            # parent_object_class is None. Question becomes "Is there an [object_class] in the room?"
            if self.object_class in [obj['objectType'] for obj in episode.get_objects()]:
                return True
            else:
                return False
        elif (self.parent_object_class in episode.receptacle_classes):
            # parent_object_class is receptacle. Question becomes "Is there an [object_class] on/in [parent_object_class]?"
            # the question is valid only when the obj/parent_obj is a valid combination for this scene
            parent_objects = [obj for obj in episode.get_objects() if obj['objectType'] == self.parent_object_class]
            for parent_object in parent_objects:
                if self.object_class in [obj.split('|')[0] for obj in parent_object['receptacleObjectIds']]:
                    return True
            return False
        else:
            raise Exception("Invalid combination: {} and {}!".format(self.object_class, self.parent_object_class))

    def __str__(self):
        """ Get the string representation of the question """
        if self.parent_object_class is None:
            return game_util.get_question_str(0, OBJECT_CLASS_TO_ID[self.object_class])
        else:
            return game_util.get_question_str(2, OBJECT_CLASS_TO_ID[self.object_class], OBJECT_CLASS_TO_ID[self.parent_object_class])

class CountQuestion(Question):
    def __init__(self, object_class, parent_object_class = None):
        super(CountQuestion, self).__init__()
        assert(parent_object_class == None or object_class.name in parent_object_class.children)

        self.object_class = object_class
        self.parent_object_class = parent_object_class
        if (parent_object_class is None) or (parent_object_class.is_openable):
            self.preposition = 'in'
        elif (parent_object_class.name == 'Box' or
              parent_object_class.name == 'GarbageCan' or
              parent_object_class.name == 'Pot' or
              parent_object_class.name == 'Sink' or
              parent_object_class.name == 'Pan'):
            self.preposition = 'in'
        else:
            self.preposition = 'on'

    def get_answer(self, episode):
        """ Get answer to the question given an episode """
        if self.parent_object_class is None:
            # parent_object_class is None. Question becomes "Is there an [object_class] in the room?"
            return len([obj for obj in episode.get_objects() if obj['objectType'] == self.object_class])
        elif self.parent_object_class.name in episode.receptacle_classes:
            total_count = 0
            # parent_object_class is receptacle. Question becomes "Is there an [object_class] on/in [parent_object_class]?"
            parent_objects = [obj for obj in episode.get_objects() if obj['objectType'] == self.parent_object_class.name]
            for parent_object in parent_objects:
                total_count += len([obj for obj in parent_object['receptacleObjectIds'] if obj.split('|')[0] == self.object_class.name])
            return total_count
        else:
            raise Exception("there is no {} in the scene!".format(self.parent_object_class.name))

    def __str__(self):
        """ Get the string representation of the question """
        if self.parent_object_class is None:
            return game_util.get_question_str(1, OBJECT_CLASS_TO_ID[self.object_class])
        else:
            return game_util.get_question_str(1, OBJECT_CLASS_TO_ID[self.object_class], OBJECT_CLASS_TO_ID[self.parent_object_class])

class ListQuestion(Question):
    def __init__(self, parent_object_class):
        super(ListQuestion, self).__init__()
        assert parent_object_class.is_receptacle, "{} is not receptacle!".format(parent_object_class.name)
        self.parent_object_class = parent_object_class
        if (parent_object_class.is_openable):
            self.preposition = 'in'
        elif (parent_object_class.name == 'Box' or
              parent_object_class.name == 'GarbageCan' or
              parent_object_class.name == 'Pot' or
              parent_object_class.name == 'Sink' or
              parent_object_class.name == 'Pan'):
            self.preposition = 'in'
        else:
            self.preposition = 'on'

    def get_answer(self, episode):
        """ Get answer to the question given an episode """
        parent_objects = [obj for obj in episode.get_objects() if obj['objectType'] == self.parent_object_class.name]
        assert len(parent_objects) == 1, "There are more than one {} in the scene. Answer is ambiguous!".format(self.parent_object_class.name)
        parent_object = parent_objects[0]
        return [obj.split('|')[0] for obj in parent_object['receptacleObjectIds']]

    def __str__(self):
        """ Get the string representation of the question """
        return("what is {} the {}").format(self.preposition, self.parent_object_class)
