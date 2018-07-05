from matchmaking_agents.agents import MatchmakingAgent
import numpy as np
import tensorflow as tf
from utils import utils

reuse = False


class DNNAgent(MatchmakingAgent):

    def __init__(self, env):
        self.input_size = env.observation_space.shape[0]
        self.create_model(env)

    def create_model(self, env):
        with tf.variable_scope("model", reuse=reuse):
            self.X = tf.placeholder(shape=[None, self.input_size, 1], dtype=env.observation_space.dtype, name="X")
            activ = tf.tanh
            conv1 = tf.layers.conv1d(
                inputs=self.X,
                filters=4,
                kernel_size=[3],
                padding="valid",
                activation=tf.nn.relu)

            conv1 = tf.reshape(conv1, [-1,  4*(self.input_size-2)])
            v_h1 = activ(self.fc(conv1, 'v_h1', nh=16, init_scale=np.sqrt(2)))
            pi_h1 = activ(self.fc(conv1, 'pi_fc1', nh=16, init_scale=np.sqrt(2)))
            self.action_probs_out = tf.nn.softmax(activ(self.fc(pi_h1, 'action_probs_out',  self.input_size + 1, init_scale=0.01, init_bias=0.0)))
            self.value = tf.reshape(activ(self.fc(v_h1, 'value', 1, init_scale=0.01, init_bias=0.0)), [-1])
        print(self.value)

        self.DISCOUNTED_REWARDS = tf.placeholder(tf.float32, [None])
        self.ADVANTAGES = tf.placeholder(tf.float32, [None])
        self.ACTIONS = tf.placeholder(tf.int32, [None, 2])

        action_indices = tf.range(tf.shape(self.ACTIONS)[0])
        prob_indices1 = tf.concat([tf.reshape(action_indices, [-1, 1]), tf.reshape(self.ACTIONS[:, 0], [-1, 1])], axis=1)
        prob_indices2 = tf.concat([tf.reshape(action_indices, [-1, 1]), tf.reshape(self.ACTIONS[:, 1], [-1, 1])], axis=1)
        prob_of_picked_action1 = tf.gather_nd(self.action_probs_out, prob_indices1)
        prob_of_picked_action2 = tf.gather_nd(self.action_probs_out, prob_indices2)
        self.prob_of_picked_action = tf.add(prob_of_picked_action1, prob_of_picked_action2)

        self.policy_loss = - tf.reduce_mean(tf.reduce_mean(self.action_probs_out) * self.ADVANTAGES)
        self.value_loss = tf.reduce_mean(tf.squared_difference(self.value, self.ADVANTAGES))
        self.loss = self.policy_loss + self.value_loss
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
        action_probs, value = self.sess.run([self.action_probs_out, self.value], {self.X: np.reshape(obs, [1, self.input_size, 1])})
        action_probs = np.reshape(action_probs, [self.input_size + 1])
        player1 = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        if player1 != self.input_size:
            removedProb = action_probs[player1] + action_probs[self.input_size]
            action_probs[player1] = 0
            action_probs[self.input_size] = 0
            for idx in range(len(action_probs)):
                if idx != player1 and idx != self.input_size:
                    action_probs[idx] += removedProb/(self.input_size-1)
            player2 = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        else:
            player2 = self.input_size
        return (player1, player2), value[0]

    def train_model(self, obs, actions, rewards, values):
        obs = np.reshape(obs, [-1, self.input_size, 1])
        actions = np.reshape(actions, [-1, 2])

        discounted_rewards, advantages, returns = utils.gae(rewards, values)

        # print(obs, action, reward)
        value_loss, policy_loss, _ = self.sess.run([self.value_loss, self.policy_loss, self.train], {self.X: obs, self.ACTIONS: actions, self.DISCOUNTED_REWARDS: discounted_rewards, self.ADVANTAGES: advantages})
        return discounted_rewards, advantages, returns, value_loss, policy_loss

    