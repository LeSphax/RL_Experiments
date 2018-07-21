import numpy as np
import math
from matchmaking_agents.Policies.policy import Policy
import random
import sys


class ScriptedPolicy(Policy):

    def __init__(self, env):
        self.input_size = env.observation_space.shape[0]
        return

    def get_action(self, obs):
        obs = list(filter(lambda a: a != -1, obs))
        # Wait until the queue is full
        if len(obs) < 2:
            return (self.input_size, self.input_size)

        # Assumes that the observation is sorted
        indexes = np.arange(len(obs))
        index1 = np.random.choice(indexes)
        indexes = np.delete(indexes, index1)
        index2 = np.random.choice(indexes)

        return (index1, index2)

    def normalize(self):
        return False