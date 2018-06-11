import glob
from utils import question_util
import h5py
import os
import random
import constants

def main():
    question_files = sorted(glob.glob('questions/*/*/*h5'))

    vocab = set()

    max_sentence_length = 0
    for file_name in question_files:
        out_file = open(os.path.splitext(file_name)[0] + '.csv', 'w')
        print('Processing file', file_name)
        out_file.write('question_type,scene_number,seed,question,answer,object_id,container_id\n')
        if 'data_existence' in file_name:
            question_type_ind = 0
        elif 'data_counting' in file_name:
            question_type_ind = 1
        elif 'data_contains' in file_name:
            question_type_ind = 2

        dataset = h5py.File(file_name)
        dataset_np = dataset['questions/question'][...]
        for line in dataset_np:
            container_ind = None
            if question_type_ind == 0:
                scene_num, scene_seed, object_ind, answer = line
                answer = str(bool(answer))
            elif question_type_ind == 1:
                scene_num, scene_seed, object_ind, answer = line
                answer = str(int(answer))
            elif question_type_ind == 2:
                scene_num, scene_seed, object_ind, container_ind, answer = line
                answer = str(bool(answer))
            question_str = question_util.get_question_str(question_type_ind, object_ind, container_ind, seed=scene_seed)
            parsed_question = question_str.replace('.', '').replace('?', '').lower().split(' ')
            max_sentence_length = max(len(parsed_question), max_sentence_length)
            vocab.update(parsed_question)

            if container_ind is None:
                container_ind = len(constants.OBJECTS)
            out_file.write('%d,%d,%d,%s,%s,%d,%d\n' % (question_type_ind, scene_num, scene_seed, question_str, answer, object_ind, container_ind))
            out_file.flush()
        print('Generated %d sentences for %s' % (dataset_np.shape[0], file_name))

    with open('vocabulary.txt', 'w') as ff:
        ff.write('\n'.join(sorted(list(vocab))))

    print('max sentence length', max_sentence_length)

if __name__ == '__main__':
    main()

