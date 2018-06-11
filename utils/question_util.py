import numpy as np
import constants
import random

def get_question_str(question_type_ind, question_object_ind, question_container_ind=None, question_direction_ind=None, seed=None):
    object_article = 'a'
    if constants.OBJECTS_SINGULAR[question_object_ind][0] in {'a', 'e', 'i', 'o', 'u'}:
        object_article = 'an'
    container_article = 'a'
    if question_container_ind is not None and constants.OBJECTS_SINGULAR[question_container_ind][0] in {'a', 'e', 'i', 'o', 'u'}:
        container_article = 'an'
    if question_container_ind is not None and constants.OBJECTS_SINGULAR[question_container_ind] in {'fridge', 'microwave', 'sink'}:
        container_article = 'the'

    if seed is not None:
        random.seed(seed)

    if question_type_ind == 0:
        template_ind = random.randint(0,5)
        if template_ind == 0:
            return 'Is there %s %s in the room?' % (object_article, constants.OBJECTS_SINGULAR[question_object_ind])
        elif template_ind == 1:
            return 'Please tell me if there is %s %s somewhere in the room.' % (object_article, constants.OBJECTS_SINGULAR[question_object_ind])
        elif template_ind == 2:
            return 'Is there %s %s somewhere in the room?' % (object_article, constants.OBJECTS_SINGULAR[question_object_ind])
        elif template_ind == 3:
            return 'Is there %s %s somewhere nearby?' % (object_article, constants.OBJECTS_SINGULAR[question_object_ind])
        elif template_ind == 4:
            return 'I think %s %s is in the room. Is that correct?' % (object_article, constants.OBJECTS_SINGULAR[question_object_ind])
        elif template_ind == 5:
            return 'Do we have any %s?' % (constants.OBJECTS_PLURAL[question_object_ind])
        else:
            raise Exception('No template')
    elif question_type_ind == 1:
        template_ind = random.randint(0,4)
        if template_ind == 0:
            return 'How many %s are there in the room?' % constants.OBJECTS_PLURAL[question_object_ind]
        elif template_ind == 1:
            return 'There are between 0 and %d %s in the room. How many are there?' % (constants.MAX_COUNTING_ANSWER, constants.OBJECTS_PLURAL[question_object_ind])
        elif template_ind == 2:
            return 'Please tell me how many %s there are somewhere in the room?' % constants.OBJECTS_PLURAL[question_object_ind]
        elif template_ind == 3:
            return 'Please tell me how many %s are around here?' % constants.OBJECTS_PLURAL[question_object_ind]
        elif template_ind == 4:
            return 'Count the number of %s in this room.' % constants.OBJECTS_PLURAL[question_object_ind]
        else:
            raise Exception('No template')
    elif question_type_ind == 2:
        preposition = 'in'
        if constants.OBJECTS[question_container_ind] in {'StoveBurner', 'TableTop'}:
            preposition = 'on'
        template_ind = random.randint(0,3)
        if template_ind == 0:
            return ('Is there %s %s %s %s %s?' % (
                object_article, constants.OBJECTS_SINGULAR[question_object_ind],
                preposition,
                container_article,
                constants.OBJECTS_SINGULAR[question_container_ind]))
        elif template_ind == 1:
            return ('Can you find %s %s %s %s %s?' % (
                object_article, constants.OBJECTS_SINGULAR[question_object_ind],
                preposition,
                container_article,
                constants.OBJECTS_SINGULAR[question_container_ind]))
        elif template_ind == 2:
            return ('Please tell me if there is %s %s %s %s %s?' % (
                object_article, constants.OBJECTS_SINGULAR[question_object_ind],
                preposition,
                container_article,
                constants.OBJECTS_SINGULAR[question_container_ind]))
        elif template_ind == 3:
            return ('I think there is %s %s %s %s %s. Is that correct?' % (
                object_article, constants.OBJECTS_SINGULAR[question_object_ind],
                preposition,
                container_article,
                constants.OBJECTS_SINGULAR[question_container_ind]))
        else:
            raise Exception('No template')

