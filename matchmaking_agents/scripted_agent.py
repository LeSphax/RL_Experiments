import math
import matchmaking_agents.agents
from matchmaking_agents.agents import MatchmakingAgent
import random
import sys


class ScriptedAgent(MatchmakingAgent):

    def __init__(self, env):
        self.input_size = env.observation_space.shape[0]
        return

    def get_action_and_value(self, obs):
        obs = list(filter(lambda a: a != -1, obs))
        # Wait until the queue is full
        if len(obs) < 2:
            return (self.input_size, self.input_size), 0

        # Assumes that the observation is sorted
        minimum_diff = sys.maxsize
        idx_min = -1
        for idx in range(len(obs) -1 ):
            diff = abs(obs[idx+1] - obs[idx])
            if minimum_diff > diff:
                minimum_diff = diff
                idx_min = idx

        return (idx_min+1, idx_min), 0
