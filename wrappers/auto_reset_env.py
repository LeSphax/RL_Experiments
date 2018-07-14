import gym
from gym.core import Wrapper

class AutoResetEnv(Wrapper):
    def __init__(self, env):
        Wrapper.__init__(self, env=env)
    
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done:
            obs = self.reset()
        return obs, rew, done, info
    
    def reset(self):
        return self.env.reset()