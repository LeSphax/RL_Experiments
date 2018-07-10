import math
import matchmaking_agents.agents
from matchmaking_agents.Values.value import MatchmakingValue
import tensorflow as tf
import numpy as np


class DNNValue(MatchmakingValue):

    def __init__(self, env, num_layers=1):
        self.input_size = env.observation_space.shape[0]
        self.create_model(env, num_layers)

    def create_model(self, env, num_layers):
        with tf.variable_scope("value", reuse=False):
            self.X = tf.placeholder(shape=[None, self.input_size], dtype=env.observation_space.dtype, name="X")

            previous_layer = self.X
            for idx in range(num_layers):
                hidden_layer = tf.contrib.layers.fully_connected(
                    inputs=previous_layer,
                    num_outputs=16,
                    activation_fn=tf.nn.tanh,
                    weights_initializer=tf.constant_initializer(np.sqrt(2))
                )
                previous_layer = hidden_layer

            self.value = tf.squeeze(tf.contrib.layers.fully_connected(
                inputs=self.X,
                num_outputs= 1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer
            ))

        self.RETURNS = tf.placeholder(tf.float32, [None])

        self.loss = tf.reduce_mean(tf.squared_difference(self.value, self.RETURNS))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        self.train = optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)


    def get_value(self,obs):
        # print(obs)
        return self.sess.run(self.value, {self.X: np.reshape(obs, [1, self.input_size])})

    def train_model(self, obs, returns):
        values, loss, _ = self.sess.run([self.value, self.loss, self.train], {self.X: obs, self.RETURNS: returns})
        return loss