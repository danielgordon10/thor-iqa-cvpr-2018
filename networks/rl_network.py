import tensorflow as tf

import constants


# Taken from https://github.com/miyosuda/async_deep_reinforce/blob/master/game_ac_network.py
class RLNetwork(object):
    def sync_from(self, src_vars, dst_vars):
        sync_ops = []
        with tf.name_scope('Sync'):
            for(src_var, dst_var) in zip(src_vars, dst_vars):
                sync_op = tf.assign(dst_var, src_var)
                sync_ops.append(sync_op)
        return tf.group(*sync_ops)


class A3CNetwork(RLNetwork):
    def rl_loss(self):
        with tf.variable_scope('a3c_loss'):
            action_size = self.pi.get_shape().as_list()[1]
            self.taken_action = tf.placeholder(tf.float32, [None, action_size], name='taken_action')

            # temporary difference (R-V) (input for policy)
            self.td = tf.placeholder(tf.float32, [None], name='td_placeholder')

            # avoid NaN with clipping when value in pi becomes zero
            log_pi = tf.log(tf.clip_by_value(self.pi, 1e-20, 1.0))

            # policy entropy
            entropy = -tf.reduce_sum(self.pi * log_pi, axis=1)

            # policy loss (output)  (Adding minus, because the original paper's
            # objective function is for gradient ascent, but we use gradient
            # descent optimizer.)
            self.policy_loss = -tf.reduce_mean(tf.reduce_sum(
                tf.multiply(log_pi, self.taken_action), axis=1) * self.td + entropy * constants.ENTROPY_BETA)

            # R (input for value)
            self.r = tf.placeholder(tf.float32, [None], name='reward_placeholder')

            # value loss (output)
            # (Learning rate for Critic is half of Actor's, so multiply by 0.5) and half from L2 Loss.
            self.value_loss = 0.25 * tf.losses.huber_loss(self.r, self.v)

            # gradienet of policy and value are summed up
            self.rl_total_loss = self.policy_loss + self.value_loss


class DeepQNetwork(RLNetwork):
    def rl_loss(self):
        with tf.variable_scope('q_loss'):
            # R (input for value)
            self.r = tf.placeholder(tf.float32, [None], name='reward_placeholder')
            action_size = self.pi.get_shape().as_list()[1]
            self.taken_action = tf.placeholder(tf.float32, [None, action_size], name='taken_action')

            q_loss = tf.losses.huber_loss(self.r, tf.reduce_sum(self.pi * self.taken_action, axis=1))
            self.rl_total_loss = tf.clip_by_value(q_loss, -10, 10)


