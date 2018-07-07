import math
import matchmaking_agents.agents
from matchmaking_agents.Values.value import MatchmakingValue
import random

class RandomValue(MatchmakingValue):

    def get_value(self,obs):
        return random.random()