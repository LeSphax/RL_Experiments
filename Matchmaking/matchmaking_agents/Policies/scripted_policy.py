import math
from matchmaking_agents.Policies.policy import Policy
import random
import sys


class ScriptedPolicy(Policy):

    def __init__(self, env, smart=True):
        self.input_size = env.action_space.n
        if smart:
            self.get_action = self.smart
        else:
            self.get_action = self.not_so_smart
        return

    def smart(self, obs):
        room = obs[0]
        obs = list(filter(lambda a: a != -1, obs[1:]))
        # Wait until the queue is full
        if len(obs) < 1:
            print("Wait ", self.input_size-1)
            return self.input_size-1, 0
        else:
            if room != -1:
                minimum_diff = sys.maxsize
                idx_min = -1
                for idx in range(len(obs)):
                    diff = abs(room - obs[idx])
                    if minimum_diff > diff:
                        minimum_diff = diff
                        idx_min = idx
                print ("take min ", idx_min)
                return idx_min, 0
            else:
                print("Take first ", 0)
                return 0, 0

    def not_so_smart(self, obs):
        obs = list(filter(lambda a: a != -1, obs))
        # Wait until the queue is full
        if len(obs) < 1:
            return self.input_size-1, 0
        else:
            return 0, 0