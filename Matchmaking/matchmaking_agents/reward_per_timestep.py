class RewardPerTimestep(object):

    def __init__(self):
        self.nb_timesteps = 0
        self.average = 0
    
    def add_reward(self, reward):
        self.nb_timesteps += 1
        self.average = self.average + (reward - self.average) / self.nb_timesteps