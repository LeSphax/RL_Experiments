import gym
from gym.core import Wrapper
import numpy as np

class NormalizeEnv(Wrapper):

    def __init__(self, env, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        Wrapper.__init__(self, env=env)
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(1)
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        obs, rews, done, infos = self.env.step(action)
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        return obs, rews[0], done, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        obs = self.env.reset()
        return self._obfilt(obs)

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        # print("Mean", self.mean)
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, values):
        delta = values - self.mean
        tot_count = self.count + 1

        new_mean = self.mean + delta / tot_count        
        m_a = self.var * (self.count)
        M2 = m_a + np.square(delta) * self.count / (tot_count)
        new_var = M2 / (tot_count)

        new_count = tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count    
