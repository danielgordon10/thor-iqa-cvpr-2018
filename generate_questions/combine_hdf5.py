import numpy as np
import h5py
import glob
import os


def combine(dataset_type, question_type):
    prefix = 'questions/'
    location = prefix + dataset_type + '/data_' + question_type + '/'

    file_name = location + 'combined.h5'
    if os.path.exists(file_name):
        os.rename(file_name, location + 'combined_old.h5')

    # Get info on all the separate datasets
    files = [ff for ff in sorted(glob.glob(location + '*.h5'))]
    good_inds_per_file = []
    good_files = []
    for file_name in files:
        try:
            h5 = h5py.File(file_name, 'r+')
            seeds = h5['questions/question'][:,1]
            good_inds = np.where(seeds != 0)[0]

            if len(good_inds) > 0:
                good_inds_per_file.append(good_inds)
                good_files.append(file_name)
            print(file_name, len(good_inds))
        except:
            print(file_name)
            print('Failed on this file')

    all_keys = []
    for key in h5.keys():
        if type(h5[key]) == h5py._hl.group.Group:
            all_keys.extend([key + '/' + k2 for k2 in h5[key].keys()])
        else:
            all_keys.append(key)

    file_name = location + 'combined.h5'
    h5_combined = h5py.File(file_name, 'w-')

    # Create all the new datasets
    row_limit = np.cumsum([len(rows) for rows in good_inds_per_file])
    row_limit = np.insert(row_limit, 0, 0)
    for key in all_keys:
        new_shape = list(h5[key].shape)
        new_shape[0] = row_limit[-1]
        h5_combined.create_dataset(key, new_shape, h5[key].dtype)

    for (ii, file_name) in enumerate(good_files):
        h5 = h5py.File(file_name, 'r')
        for key in all_keys:
            h5_combined[key][row_limit[ii]:row_limit[ii + 1], ...] = h5[key][good_inds_per_file[ii], ...]

    for key in all_keys:
        print('Key : ' + key + ' shape ' + str(h5_combined[key].shape))

    for file_name in good_files:
        os.remove(file_name)


if __name__ == '__main__':
    for dataset_type in ['train', 'test', 'train_test']:
        for question_type in ['existence', 'contains', 'counting']:
            combine(dataset_type, question_type)
