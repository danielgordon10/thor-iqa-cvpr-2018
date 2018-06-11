import tensorflow as tf
import constants
slim = tf.contrib.slim


class BatchGraphGRUCell(tf.contrib.rnn.RNNCell):

    def __init__(self, initializer=None, activation=tf.nn.tanh, reuse=None):

        self._initializer = initializer
        self._activation = activation
        self._reuse = reuse

        self._output_size = constants.MEMORY_SIZE * constants.SPATIAL_MAP_WIDTH * constants.SPATIAL_MAP_HEIGHT
        self._patch_size = constants.MEMORY_SIZE * constants.STEPS_AHEAD * constants.STEPS_AHEAD

    @property
    def state_size(self):
        return self._output_size, self._patch_size

    @property
    def output_size(self):
        return self._output_size, self._patch_size

    def get_memory_patch(self, pose, memory):
        # Shift offsets to global coordinate frame.
        xMin = tf.case({
                tf.equal(pose[2], 0): lambda: pose[0] - int(constants.STEPS_AHEAD / 2),
                tf.equal(pose[2], 1): lambda: pose[0] + 1,
                tf.equal(pose[2], 2): lambda: pose[0] - int(constants.STEPS_AHEAD / 2),
                tf.equal(pose[2], 3): lambda: pose[0] - constants.STEPS_AHEAD},
                default=lambda: 0,
                exclusive=True)

        yMin = tf.case({
                tf.equal(pose[2], 0): lambda: pose[1] + 1,
                tf.equal(pose[2], 1): lambda: pose[1] - int(constants.STEPS_AHEAD / 2),
                tf.equal(pose[2], 2): lambda: pose[1] - constants.STEPS_AHEAD,
                tf.equal(pose[2], 3): lambda: pose[1] - int(constants.STEPS_AHEAD / 2)},
                default=lambda: 0,
                exclusive=True)

        graph_patch = memory[yMin:yMin + constants.STEPS_AHEAD,
                xMin:xMin + constants.STEPS_AHEAD, :]
        graph_patch = tf.reshape(graph_patch, [constants.STEPS_AHEAD, constants.STEPS_AHEAD, constants.MEMORY_SIZE])

        graph_patch = tf.cond(tf.equal(pose[2], 0),
                lambda: graph_patch,
                lambda: tf.image.rot90(graph_patch, (-pose[2] % 4)))
        return graph_patch

    def update_memory_patch(self, pose, graph_patch, memory):
        graph_patch = tf.cond(tf.equal(pose[2], 0),
                lambda: graph_patch,
                lambda: tf.image.rot90(graph_patch, pose[2]))

        xMin = tf.case({
                tf.equal(pose[2], 0): lambda: pose[0] - int(constants.STEPS_AHEAD / 2),
                tf.equal(pose[2], 1): lambda: pose[0] + 1,
                tf.equal(pose[2], 2): lambda: pose[0] - int(constants.STEPS_AHEAD / 2),
                tf.equal(pose[2], 3): lambda: pose[0] - constants.STEPS_AHEAD},
                default=lambda: 0,
                exclusive=True)

        yMin = tf.case({
                tf.equal(pose[2], 0): lambda: pose[1] + 1,
                tf.equal(pose[2], 1): lambda: pose[1] - int(constants.STEPS_AHEAD / 2),
                tf.equal(pose[2], 2): lambda: pose[1] - constants.STEPS_AHEAD,
                tf.equal(pose[2], 3): lambda: pose[1] - int(constants.STEPS_AHEAD / 2)},
                default=lambda: 0,
                exclusive=True)

        # Now the ugliest slicing ever.
        top_slice = memory[:yMin, :, :]
        bottom_slice = memory[yMin + constants.STEPS_AHEAD:, :, :]
        left_slice = memory[yMin:yMin + constants.STEPS_AHEAD, :xMin, :]
        right_slice = memory[yMin:yMin + constants.STEPS_AHEAD,xMin + constants.STEPS_AHEAD:, :]

        memory = tf.concat((left_slice, graph_patch, right_slice), axis=1)
        memory = tf.concat((top_slice, memory, bottom_slice), axis=0)
        memory = tf.reshape(memory, [constants.SPATIAL_MAP_HEIGHT, constants.SPATIAL_MAP_WIDTH, constants.MEMORY_SIZE])
        return memory

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope('GraphGRU', reuse=self._reuse):
            inputs, pose = inputs
            state, _ = state
            batch_size = inputs.get_shape().as_list()[0]
            memory = tf.reshape(state,
                    [batch_size, constants.SPATIAL_MAP_HEIGHT, constants.SPATIAL_MAP_WIDTH, constants.MEMORY_SIZE])
            memory_chunks = tf.unstack(memory, axis=0)

            pose_chunks = tf.unstack(pose, axis=0)
            state_patches = []
            for uu in range(batch_size):
                state_patches.append(self.get_memory_patch(pose_chunks[uu], memory_chunks[uu]))

            state_patch = tf.stack(state_patches, axis=0)
            state_patch_shape = state_patch.get_shape()
            state_patch = tf.reshape(state_patch, (batch_size, -1))
            state_concat = tf.concat((inputs, state_patch), axis=-1)

            update_gate = slim.fully_connected(state_concat, self._patch_size,
                    scope='update_gate', activation_fn=tf.nn.sigmoid)
            reset_gate = slim.fully_connected(state_concat, self._patch_size,
                    scope='reset_gate', activation_fn=tf.nn.sigmoid)
            h_tilde = tf.nn.tanh(
                    slim.fully_connected(reset_gate * state_patch, self._patch_size,
                        scope='h_tilde_s', activation_fn=None) +
                    slim.fully_connected(inputs, self._patch_size,
                        scope='h_tilde_i', activation_fn=None))
            output_patch = update_gate * h_tilde + (1 - update_gate) * state_patch

            output_patch = tf.reshape(output_patch, state_patch_shape)

            output_memory = []
            output_patches = tf.unstack(output_patch, axis=0)
            for uu in range(batch_size):
                output_memory.append(self.update_memory_patch(pose_chunks[uu], output_patches[uu], memory_chunks[uu]))
            output_memory = tf.concat(output_memory, axis=0)

            output_memory = tf.reshape(output_memory, [batch_size, -1])
            output_patch = tf.reshape(output_patch, [batch_size, -1])
            return (output_memory, output_patch), (output_memory, output_patch)
