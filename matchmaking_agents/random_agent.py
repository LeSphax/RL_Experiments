import math
import matchmaking_agents.agents
from matchmaking_agents.agents import MatchmakingAgent
import random

class RandomAgent(MatchmakingAgent):

    def get_action_and_value(self,obs):
        return (math.floor(random.random() * len(obs)), math.floor(random.random() *len(obs)))