import math
import matchmaking_agents.agents
from matchmaking_agents.Policies.policy import MatchmakingPolicy
import tensorflow as tf
import numpy as np


class DNNPolicy(MatchmakingPolicy):

    def __init__(self, env, num_layers = 1):
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
                    num_outputs=16,
                    activation_fn=tf.nn.relu,
                    weights_initializer=tf.zeros_initializer
                )
                previous_layer = hidden_layer

            self.output_layer = tf.squeeze(tf.contrib.layers.fully_connected(
                inputs=previous_layer,
                num_outputs=self.output_size,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer
            ))

            self.action_probs = tf.nn.softmax(self.output_layer)

        self.ADVANTAGES = tf.placeholder(tf.float32, [None])
        self.ACTIONS = tf.placeholder(tf.int32, [None])

        self.action_indices = tf.range(tf.shape(self.ACTIONS)[0])
        self.prob_indices = tf.concat([tf.reshape(self.action_indices, [-1, 1]), tf.reshape(self.ACTIONS, [-1, 1])], axis=1)
        self.prob_of_picked_action = tf.gather_nd(self.action_probs, self.prob_indices)

        self.entropy = - tf.reduce_sum(self.action_probs * tf.log(self.action_probs), 1, name="entropy")
        self.entropy_mean = tf.reduce_mean(self.entropy)

        self.losses = - tf.log(self.prob_of_picked_action) * self.ADVANTAGES - self.entropy * 0.01
        self.loss = tf.reduce_mean(self.losses)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train = optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def get_action(self, obs):
        action_probs = self.sess.run(self.action_probs, {self.X: np.reshape(obs, [1, self.input_size])})

        return np.random.choice(np.arange(len(action_probs)), p=action_probs)

    def train_model(self, obs, actions, advantages):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        entropy, loss, _ = self.sess.run([self.entropy_mean, self.loss, self.train], {self.X: obs, self.ACTIONS: actions, self.ADVANTAGES: advantages})
        return entropy, loss
