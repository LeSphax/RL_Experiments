import math
from matchmaking_agents.Policies.policy import Policy
import tensorflow as tf
import numpy as np


class CategoricalPd(object):
    def __init__(self, logits):
        self.logits = logits
    def neglogp(self, x):
        one_hot_actions = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
        return tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=one_hot_actions)
    def entropy(self):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keep_dims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keep_dims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)
    def sample(self):
        u = tf.random_uniform(tf.shape(self.logits))
        return tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)

class DNNPolicy(Policy):

    def __init__(self, env, num_layers=1):
        self.input_size = env.observation_space.shape[0]
        self.output_size = env.action_space.n
        self.create_model(env, num_layers)

    def create_model(self, env, num_layers):
        with tf.variable_scope("policy", reuse=False):
            self.X = tf.placeholder(shape=[None, self.input_size], dtype=env.observation_space.dtype, name="X")

            previous_layer = self.X
            for idx in range(num_layers):
                hidden_layer = tf.contrib.layers.fully_connected(
                    inputs=previous_layer,
                    num_outputs=64,
                    activation_fn=tf.nn.tanh,
                    weights_initializer=tf.constant_initializer(np.sqrt(2))
                )
                previous_layer = hidden_layer

            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=previous_layer,
                num_outputs=self.output_size,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer
            )

            self.probability_distribution = CategoricalPd(self.output_layer)

            self.action = self.probability_distribution.sample()
            self.neglogp_action = self.probability_distribution.neglogp(self.action)

        self.ADVANTAGES = tf.placeholder(tf.float32, [None])
        self.ACTIONS = tf.placeholder(tf.int32, [None])
        self.OLDNEGLOGP_ACTIONS = tf.placeholder(tf.float32, [None])
        self.LEARNING_RATE = tf.placeholder(tf.float32, ())
        self.CLIPPING = tf.placeholder(tf.float32, ())

        self.new_neglogp_action = self.probability_distribution.neglogp(self.ACTIONS)

        self.entropy = tf.reduce_mean(self.probability_distribution.entropy())

        ratio = tf.exp(self.OLDNEGLOGP_ACTIONS - self.new_neglogp_action)
        pg_losses = -self.ADVANTAGES * ratio
        pg_losses2 = -self.ADVANTAGES * tf.clip_by_value(ratio, 1.0 - self.CLIPPING, 1.0 + self.CLIPPING)
        self.loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2)) - self.entropy * 0.01
        optimizer = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE)
        self.train = optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def get_action(self, obs):
        action, neglogp_action = self.sess.run([self.action, self.neglogp_action], {self.X: np.reshape(obs, [1, self.input_size])})

        return action[0], neglogp_action[0]

    def train_model(self, obs, actions, neglogp_actions, advantages, clipping, learning_rate):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        entropy, loss, _ = self.sess.run(
            [self.entropy, self.loss, self.train],
            {
                self.X: obs,
                self.ACTIONS: actions,
                self.OLDNEGLOGP_ACTIONS: neglogp_actions,
                self.ADVANTAGES: advantages,
                self.CLIPPING: clipping,
                self.LEARNING_RATE: learning_rate,
            }
        )
        return entropy, loss
