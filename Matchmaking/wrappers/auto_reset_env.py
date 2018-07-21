import gym
from gym.core import Wrapper


class AutoResetEnv(Wrapper):
    def __init__(self, env, max_steps=None):
        Wrapper.__init__(self, env=env)
        self.steps = 0
        self.max_steps = max_steps

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.steps += 1
        if self.max_steps is not None and self.steps >= self.max_steps:
            done = True
        if done:
            obs = self.reset()
        return obs, rew, done, info

    def reset(self):
        self.steps = 0
        return self.env.reset()
