import numpy as np
import tensorflow as tf
import os


def kernel_to_image(data, padsize=1, padval=0):
    # Turns a convolutional kernel into an image of nicely tiled filters.
    # Useful for viewing purposes.
    if len(data.get_shape().as_list()) > 4:
        data = tf.squeeze(data)
    data = tf.transpose(data, (3, 0, 1, 2))
    data_shape = tuple(data.get_shape().as_list())
    min_val = tf.reduce_min(tf.reshape(data, (data_shape[0], -1)), reduction_indices=1)
    data = tf.transpose((tf.transpose(data, (1, 2, 3, 0)) - min_val), (3, 0, 1, 2))
    max_val = tf.reduce_max(tf.reshape(data, (data_shape[0], -1)), reduction_indices=1)
    data = tf.transpose((tf.transpose(data, (1, 2, 3, 0)) / max_val), (3, 0, 1, 2))

    n = int(np.ceil(np.sqrt(data_shape[0])))
    ndim = data.get_shape().ndims
    padding = ((0, n ** 2 - data_shape[0]), (0, padsize),
            (0, padsize)) + ((0, 0),) * (ndim - 3)
    data = tf.pad(data, padding, mode='constant')
    # tile the filters into an image
    data_shape = tuple(data.get_shape().as_list())
    data = tf.transpose(tf.reshape(data, ((n, n) + data_shape[1:])), ((0, 2, 1, 3)
            + tuple(range(4, ndim + 1))))
    data_shape = tuple(data.get_shape().as_list())
    data = tf.reshape(data, ((n * data_shape[1], n * data_shape[3]) + data_shape[4:]))
    return tf.image.convert_image_dtype(data, dtype=tf.uint8)


class empty_scope():
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        pass


def cond_scope(scope):
    return empty_scope() if scope is None else tf.variable_scope(scope)


def variable_summaries(var, scope=''):
    # Some useful stats for variables.
    if len(scope) > 0:
        scope = '/' + scope
    with tf.name_scope('summaries' + scope):
        mean = tf.reduce_mean(tf.abs(var))
        with tf.device('/cpu:0'):
            return tf.summary.scalar('mean_abs', mean)


def conv_variable_summaries(var, scope=''):
    # Useful stats for variables and the kernel images.
    var_summary = variable_summaries(var, scope)
    if len(scope) > 0:
        scope = '/' + scope
    with tf.name_scope('conv_summaries' + scope):
        var_shape = var.get_shape().as_list()
        if not(var_shape[0] == 1 and var_shape[1] == 1):
            if var_shape[2] < 3:
                var = tf.tile(var, [1, 1, 3, 1])
                var_shape = var.get_shape().as_list()
            summary_image = tf.expand_dims(
                    kernel_to_image(tf.slice(
                        var, [0, 0, 0, 0], [var_shape[0], var_shape[1], 3, var_shape[3]])),
                    0)
            with tf.device('/cpu:0'):
                image_summary = tf.summary.image('filters', summary_image)
                var_summary = tf.summary.merge([var_summary, image_summary])
    return var_summary


def restore(session, save_file, raise_if_not_found=False):
    if not os.path.exists(save_file) and raise_if_not_found:
        raise Exception('File %s not found' % save_file)
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
            if var.name.split(':')[0] in saved_shapes])
    var_name_to_var = {var.name : var for var in tf.global_variables()}
    restore_vars = []
    restored_var_names = set()
    restored_var_new_shape = []
    print('Restoring:')
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        for var_name, saved_var_name in var_names:
            if 'global_step' in var_name:
                restored_var_names.add(saved_var_name)
                continue
            curr_var = var_name_to_var[var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
                print(str(saved_var_name) + ' -> \t' + str(var_shape) + ' = ' +
                      str(int(np.prod(var_shape) * 4 / 10**6)) + 'MB')
                restored_var_names.add(saved_var_name)
            else:
                print('Shape mismatch for var', saved_var_name, 'expected', var_shape,
                      'got', saved_shapes[saved_var_name])
                restored_var_new_shape.append((saved_var_name, curr_var, reader.get_tensor(saved_var_name)))
                print('bad things')
    ignored_var_names = sorted(list(set(saved_shapes.keys()) - restored_var_names))
    print('\n')
    if len(ignored_var_names) == 0:
        print('Restored all variables')
    else:
        print('Did not restore:' + '\n\t'.join(ignored_var_names))

    if len(restore_vars) > 0:
        saver = tf.train.Saver(restore_vars)
        saver.restore(session, save_file)

    if len(restored_var_new_shape) > 0:
        print('trying to restore misshapen variables')
        assign_ops = []
        for name, kk, vv in restored_var_new_shape:
            copy_sizes = np.minimum(kk.get_shape().as_list(), vv.shape)
            slices = [slice(0,cs) for cs in copy_sizes]
            print('copy shape', name, kk.get_shape().as_list(), '->', copy_sizes.tolist())
            new_arr = session.run(kk)
            new_arr[slices] = vv[slices]
            assign_ops.append(tf.assign(kk, new_arr))
        session.run(assign_ops)
        print('Copying unmatched weights done')
    print('Restored %s' % save_file)
    try:
        start_iter = int(save_file.split('-')[-1])
    except ValueError:
        print('Could not parse start iter, assuming 0')
        start_iter = 0
    return start_iter


def restore_from_dir(sess, folder_path, raise_if_not_found=False):
    start_iter = 0
    ckpt = tf.train.get_checkpoint_state(folder_path)
    if ckpt and ckpt.model_checkpoint_path:
        print('Restoring')
        start_iter = restore(sess, ckpt.model_checkpoint_path)
    else:
        if raise_if_not_found:
            raise Exception('No checkpoint to restore in %s' % folder_path)
        else:
            print('No checkpoint to restore')
    return start_iter


def save(saver, sess, folder_path, global_step):
    print('Saving...')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    saver.save(sess, os.path.join(folder_path, 'checkpoint'), global_step=global_step)
    print('Saved snapshot', os.path.join(folder_path, 'chcekpoint'), global_step)


def remove_axis_get_shape(curr_shape, axis):
    assert axis > 0, 'Axis must be greater than 0'
    axis_shape = curr_shape.pop(axis)
    curr_shape[axis - 1] *= axis_shape
    return curr_shape


def remove_axis(input_tensor, axis):
    with tf.name_scope('remove_axis'):
        tensor_shape = tf.shape(input_tensor)
        curr_shape = input_tensor.get_shape().as_list()
        curr_shape = [ss if ss is not None else tensor_shape[ii] for ii,ss in enumerate(curr_shape)]
        if type(axis) == int:
            new_shape = remove_axis_get_shape(curr_shape, axis)
        else:
            for ax in sorted(axis, reverse=True):
                new_shape = remove_axis_get_shape(curr_shape, ax)
        return tf.reshape(input_tensor, tf.stack(new_shape))


def split_axis_get_shape(curr_shape, axis, d1, d2):
    assert axis < len(curr_shape), 'Axis must be less than the current rank'
    #assert curr_shape[axis] == d1 * d2, 'Dimensions are not evenly split by shape'
    curr_shape.insert(axis, d1)
    curr_shape[axis + 1] = d2
    return curr_shape


def split_axis(input_tensor, axis, d1, d2):
    with tf.name_scope('split_axis'):
        tensor_shape = tf.shape(input_tensor)
        curr_shape = input_tensor.get_shape().as_list()
        curr_shape = [ss if ss is not None else tensor_shape[ii] for (ii, ss) in enumerate(curr_shape)]
        new_shape = split_axis_get_shape(curr_shape, axis, d1, d2)
        return tf.reshape(input_tensor, tf.stack(new_shape))


def l2_regularizer():
    with tf.variable_scope('l2_weight_penalty'):
        l2_weight_penalty = 0.0005 * tf.add_n([tf.nn.l2_loss(v)
            for v in tf.trainable_variables()])
    return l2_weight_penalty


def Session():
    return tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True), allow_soft_placement=True))
