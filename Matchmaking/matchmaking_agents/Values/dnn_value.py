from Matchmaking.matchmaking_agents.Values.value import MatchmakingValue
import tensorflow as tf
import numpy as np


class DNNValue(MatchmakingValue):

    def __init__(self, model_function, env, reuse=False):
        self.input_shape = env.observation_space.shape
        name = 'value'

        self.X, previous_layer = model_function(name, self.input_shape, reuse)

        with tf.variable_scope(name + '/training', reuse=reuse):
            self.value = tf.contrib.layers.fully_connected(
                inputs=previous_layer,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.orthogonal_initializer(1)
            )[:, 0]

            self.OLD_VALUES = tf.placeholder(tf.float32, [None], name="old_values")
            self.RETURNS = tf.placeholder(tf.float32, [None], name="returns")
            self.LEARNING_RATE = tf.placeholder(tf.float32, (), name="learning_rate")
            self.CLIPPING = tf.placeholder(tf.float32, (), name="clipping")

            value_clipped = self.OLD_VALUES + tf.clip_by_value(self.value - self.OLD_VALUES, -self.CLIPPING, self.CLIPPING)
            losses1 = tf.square(self.value - self.RETURNS)
            losses2 = tf.square(value_clipped - self.RETURNS)
            self.loss = .5 * tf.reduce_mean(tf.maximum(losses1, losses2))

            self.params = tf.trainable_variables()
            grads = tf.gradients(self.loss, self.params)
            grads, _grad_norm = tf.clip_by_global_norm(grads, 0.5)
            self.grads_and_vars = list(zip(grads, self.params))

            optimizer = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE, epsilon=1e-5)
            self._train = optimizer.apply_gradients(self.grads_and_vars)

            self.sess = tf.get_default_session()
            init = tf.global_variables_initializer()
            self.sess.run(init)

    def get_value(self, obs):
        return self.sess.run(self.value, {self.X: obs})

    def train(self, obs, values, returns, clipping, learning_rate):
        values, loss, _ = self.sess.run(
            [self.value, self.loss, self._train],
            {
                self.X: obs,
                self.OLD_VALUES: values,
                self.RETURNS: returns,
                self.CLIPPING: clipping,
                self.LEARNING_RATE: learning_rate,
            }
        )
        return loss

    def get_weights(self):
        values = tf.get_default_session().run(tf.global_variables())
        return values

    def set_weights(self, weights):
        for i, value in enumerate(weights):
            value = np.asarray(value)
            tf.global_variables()[i].load(value, self.sess)