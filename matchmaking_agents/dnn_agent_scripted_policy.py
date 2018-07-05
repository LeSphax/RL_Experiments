from matchmaking_agents.agents import MatchmakingAgent
from matchmaking_agents.scripted_agent import ScriptedAgent
from matchmaking_agents.random_agent import RandomAgent
import numpy as np
import tensorflow as tf
from utils import utils

reuse = False


class DNNScriptedAgent(MatchmakingAgent):

    def __init__(self, env):
        self.input_size = env.observation_space.shape[0]
        self.scripted_agent = ScriptedAgent(env)
        self.create_model(env)

    def create_model(self, env):
        with tf.variable_scope("model", reuse=reuse):
            self.X = tf.placeholder(shape=[None, self.input_size, 1], dtype=env.observation_space.dtype, name="X")
            activ = tf.tanh
            self.conv1 = tf.layers.conv1d(
                inputs=self.X,
                filters=4,
                kernel_size=[3],
                padding="valid",
                activation=tf.nn.relu)

            self.conv1 = tf.reshape(self.conv1, [-1,  4*(self.input_size-2)])
            v_h1 = activ(self.fc(self.conv1, 'v_h1', nh=16, init_scale=np.sqrt(2)))
            self.value = tf.squeeze(activ(self.fc(v_h1, 'value', 1, init_scale=0.01, init_bias=0.0)))

        self.DISCOUNTED_REWARDS = tf.placeholder(tf.float32, [None])
        self.ADVANTAGES = tf.placeholder(tf.float32, [None])
        self.RETURNS = tf.placeholder(tf.float32, [None])
        self.ACTIONS = tf.placeholder(tf.int32, [None, 2])

        self.value_loss = tf.reduce_mean(tf.squared_difference(self.value, self.RETURNS))
        self.loss = self.value_loss
        optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
        self.train = optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def fc(self, x, scope, nh, *, init_scale=1.0, init_bias=0.0):
        with tf.variable_scope(scope):
            nin = x.get_shape()[1].value
            w = tf.get_variable(
                "w", [nin, nh], initializer=tf.orthogonal_initializer(init_scale))
            b = tf.get_variable(
                "b", [nh], initializer=tf.constant_initializer(init_bias))
            return tf.matmul(x, w)+b

    def get_action_and_value(self, obs):
        value = self.sess.run([self.value], {self.X: np.reshape(obs, [1, self.input_size, 1])})
        players, _ = self.scripted_agent.get_action_and_value(obs)
        return players, value[0]

    def train_model(self, obs, actions, rewards, values):
        obs = np.reshape(obs, [-1, self.input_size, 1])
        actions = np.reshape(actions, [-1, 2])

        discounted_rewards, advantages, returns = utils.gae(rewards, values, 0.9)

        # sd, valuesss = self.sess.run(
        #     [tf.squared_difference(self.value, self.RETURNS), self.value],
        #     {
        #         self.X: obs,
        #         self.ACTIONS: actions,
        #         self.DISCOUNTED_REWARDS: discounted_rewards,
        #         self.ADVANTAGES: advantages,
        #         self.RETURNS: returns
        #     }
        # )
        # print(valuesss)
        # print("VALUES ", values)
        # print("RETURNS ", returns)
        # print("ADVANTAGES ", advantages)
        # print("SD ", sd)
        # print("SD ", np.sum(sd))

        value_loss, _ = self.sess.run(
            [self.value_loss, self.train],
            {
                self.X: obs,
                self.ACTIONS: actions,
                self.DISCOUNTED_REWARDS: discounted_rewards,
                self.ADVANTAGES: advantages,
                self.RETURNS: returns
            }
        )

        return discounted_rewards, advantages, returns, value_loss, 0.0
