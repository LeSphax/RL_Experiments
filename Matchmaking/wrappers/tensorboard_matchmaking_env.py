import numpy as np
import tensorflow as tf
from gym.core import Wrapper


class TensorboardMathmakingEnv(Wrapper):
    def __init__(self, env, save_path):
        Wrapper.__init__(self, env=env)
        self.reset_episode_lists()

        self.total_nb_steps = 0
        self.rewards = None
        self.actions = None

        self.TOTAL_REWARD = tf.placeholder(tf.float32, ())
        self.HOLDS = tf.placeholder(tf.float32, ())
        self.PUNISHES = tf.placeholder(tf.float32, ())

        tf.summary.scalar('total_reward', self.TOTAL_REWARD)
        tf.summary.scalar('holds', self.HOLDS)
        tf.summary.scalar('punishes', self.PUNISHES)

        self.merged = tf.summary.merge_all()
        self.sess = tf.get_default_session()
        self.train_writer = tf.summary.FileWriter(save_path, self.sess.graph)

    def reset_episode_lists(self):
        self.episode_rewards = []
        self.episode_punishes = []
        self.episode_holds = []

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.rewards.append(rew)
        self.actions.append(action)
        self.total_nb_steps += 1
        if self.total_nb_steps % 1028 == 0:
            summary = self.sess.run(
                self.merged,
                {
                    self.TOTAL_REWARD: np.mean(self.episode_rewards),
                    self.HOLDS: np.mean(self.episode_holds),
                    self.PUNISHES: np.mean(self.episode_punishes),
                }
            )
            self.train_writer.add_summary(summary, self.total_nb_steps)
            self.reset_episode_lists()

        return obs, rew, done, info

    def reset(self):
        if self.rewards is not None:
            self.episode_rewards.append(np.sum(self.rewards))

            unique, counts = np.unique(self.rewards, return_counts=True)
            reward_values = dict(zip(unique, counts))
            self.episode_punishes.append(reward_values.get(-0.1, 0))


            unique, counts = np.unique(self.actions, return_counts=True)
            actions_values = dict(zip(unique, counts))
            self.episode_holds.append(actions_values.get(self.env.action_space.n - 1, 0))

        self.rewards = []
        self.actions = []
        return self.env.reset()
