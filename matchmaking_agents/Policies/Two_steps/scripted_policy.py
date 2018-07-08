import math
import matchmaking_agents.agents
from matchmaking_agents.Policies.policy import MatchmakingPolicy
import random
import sys


class ScriptedPolicy(MatchmakingPolicy):

    def __init__(self, env):
        self.input_size = env.action_space.n
        return

    def get_action(self, obs):
        obs = list(filter(lambda a: a != -1, obs))
        # Wait until the queue is full
        if len(obs) < 1:
            return self.input_size-1
        else:
            return 0
