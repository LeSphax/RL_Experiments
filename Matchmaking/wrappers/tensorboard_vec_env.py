import numpy as np
import tensorflow as tf
import time

from Matchmaking.wrappers.vec_env import VecEnvWrapper
from gym.core import Wrapper


class TensorboardVecEnv(VecEnvWrapper):
    def __init__(self, venv, save_path, summary_interval=1028, *, reward_key='total_reward'):
        VecEnvWrapper.__init__(self, venv=venv)
        self.reward_key = reward_key
        self.epinfos = []
        self.total_nb_steps = 0
        # Need summary_interval % num_env = 0
        self.summary_interval = summary_interval * self.num_envs

        self.save_path = save_path

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

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.total_nb_steps += self.num_envs
        for info in infos:
            maybeepinfo = info.get('episode')
            if maybeepinfo: self.epinfos.append(maybeepinfo)

        if self.total_nb_steps % self.summary_interval == 0:
            eprewards = [epinfo[self.reward_key] for epinfo in self.epinfos]
            print(eprewards)

            mean_reward = np.mean(eprewards) if len(eprewards) > 0 else 0
            fps = self.summary_interval / (time.time() - self.previous_summary_time)

            print("Sum: ", mean_reward, fps, self.total_nb_steps)
            summary = self.sess.run(
                self.merged,
                {
                    self.TOTAL_REWARD: mean_reward,
                    self.FPS: fps
                })
            self.train_writer.add_summary(summary, self.total_nb_steps)
            self.epinfos = []
            self.previous_summary_time = time.time()

        return obs, rews, news, infos

    def reset(self):
        return self.venv.reset()
