import math
from matchmaking_agents.Policies.policy import MatchmakingPolicy
import random

class RandomPolicy(MatchmakingPolicy):

    def get_action(self,obs):
        return math.floor(random.random() * len(obs))