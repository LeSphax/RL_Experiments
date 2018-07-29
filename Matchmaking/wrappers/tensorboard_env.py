import numpy as np
import tensorflow as tf
import time
from gym.core import Wrapper


class TensorboardEnv(Wrapper):
    def __init__(self, env, save_path, summary_interval=1028):
        Wrapper.__init__(self, env=env)
        self.episode_rewards = []
        self.total_nb_steps = 0
        self.rewards = None
        self.save_path = save_path

        self.summary_interval = summary_interval

        self.create_summaries(save_path)

        self.previous_summary_time = time.time()

    def create_summaries(self, save_path):
        self.TOTAL_REWARD = tf.placeholder(tf.float32, ())
        self.FPS = tf.placeholder(tf.float32, ())
        tf.summary.scalar('total_reward', self.TOTAL_REWARD)
        tf.summary.scalar('fps', self.FPS)
        self.merged = tf.summary.merge_all()
        self.sess = tf.get_default_session()
        self.train_writer = tf.summary.FileWriter(save_path)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.rewards.append(rew)
        self.total_nb_steps += 1
        if self.total_nb_steps % self.summary_interval == 0:

            print("Create dem summaries", np.mean(self.episode_rewards),
                  self.summary_interval / (time.time() - self.previous_summary_time))
            summary = self.sess.run(self.merged, {
                self.TOTAL_REWARD: np.mean(self.episode_rewards),
                self.FPS: self.summary_interval / (time.time() - self.previous_summary_time)
            })
            self.train_writer.add_summary(summary, self.total_nb_steps)
            self.episode_rewards = []
            self.previous_summary_time = time.time()

        return obs, rew, done, info

    def reset(self):
        if self.rewards is not None:
            self.episode_rewards.append(np.sum(self.rewards))
        self.rewards = []
        return self.env.reset()
