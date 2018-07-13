import gym
from gym.core import Wrapper
import numpy as np

class MonitorEnv(Wrapper):
    def __init__(self, env):
        Wrapper.__init__(self, env=env)
    
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.steps += 1
        self.rewards.append(rew)
        if done:
            info['episode'] = {
                'total_reward': np.sum(self.rewards),
                'nb_steps': self.steps
            }
        return obs, rew, done, info

    def reset(self):
        self.steps = 0
        self.rewards = []
        return self.env.reset()

    def close(self):
        return

    def render(self):
        return self.env.render()