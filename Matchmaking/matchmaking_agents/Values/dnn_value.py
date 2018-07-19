from Matchmaking.matchmaking_agents.Values.value import MatchmakingValue
import tensorflow as tf
import numpy as np
from Matchmaking.matchmaking_agents.model import create_model


class DNNValue(MatchmakingValue):

    def __init__(self, env, num_layers, num_conv_layers):
        self.input_size = env.observation_space.shape[0]

        self.X, previous_layer = create_model("value", self.input_size, num_layers, num_conv_layers)

        self.value = tf.contrib.layers.fully_connected(
                inputs=previous_layer,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.constant_initializer(1)
            )[:,0]


        self.OLD_VALUES = tf.placeholder(tf.float32, [None])
        self.RETURNS = tf.placeholder(tf.float32, [None])
        self.LEARNING_RATE = tf.placeholder(tf.float32, ())
        self.CLIPPING = tf.placeholder(tf.float32, ())

        value_clipped = self.OLD_VALUES + tf.clip_by_value(self.value - self.OLD_VALUES, -self.CLIPPING,  self.CLIPPING)
        losses1 = tf.square(self.value - self.RETURNS)
        losses2 = tf.square(value_clipped - self.RETURNS)
        self.loss = .5 * tf.reduce_mean(tf.maximum(losses1, losses2))

        with tf.variable_scope('value'):
            params = tf.trainable_variables()
        grads = tf.gradients(self.loss, params)
        grads, _grad_norm = tf.clip_by_global_norm(grads, 0.5)
        grads = list(zip(grads, params))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE, epsilon=1e-5)
        self._train = optimizer.apply_gradients(grads)

        self.sess = tf.get_default_session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def get_value(self, obs):
        # print(obs)
        return self.sess.run(self.value, {self.X: np.reshape(obs, [1, self.input_size])})

    def train_model(self, obs, values, returns, clipping, learning_rate):
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
