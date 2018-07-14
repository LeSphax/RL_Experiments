import gym
from gym.core import Wrapper
import numpy as np
import tensorflow as tf
from datetime import datetime

class TensorboardEnv(Wrapper):
    def __init__(self, env, write=True):
        Wrapper.__init__(self, env=env)
        self.episode_rewards = []
        self.total_nb_steps = 0
        self.rewards = None

        name = datetime.now().strftime("%Y%m%d-%H%M%S")

        self.TOTAL_REWARD = tf.placeholder(tf.float32, ())

        tf.summary.scalar('total_reward', self.TOTAL_REWARD)

        self.merged = tf.summary.merge_all()
        self.sess = tf.Session()
        self.train_writer = tf.summary.FileWriter('../../../Experiments/Matchmaking/train/{name}'.format(name=name), self.sess.graph)


    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.rewards.append(rew)
        self.total_nb_steps += 1
        if self.total_nb_steps % 1028 == 0:
            summary = self.sess.run(self.merged, {self.TOTAL_REWARD: np.mean(self.episode_rewards)})
            self.train_writer.add_summary(summary, self.total_nb_steps)
            self.episode_rewards = []

        return obs, rew, done, info

    def reset(self):
        if self.rewards is not None:
            self.episode_rewards.append(np.sum(self.rewards))
        self.rewards = []
        return self.env.reset()