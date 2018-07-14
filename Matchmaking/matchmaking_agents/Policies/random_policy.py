import math
from matchmaking_agents.Policies.policy import Policy
import random

class RandomPolicy(Policy):

    def get_action(self,obs):
        return math.floor(random.random() * len(obs))