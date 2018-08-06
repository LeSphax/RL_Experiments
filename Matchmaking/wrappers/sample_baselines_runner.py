import gym
from gym.core import Wrapper
import numpy as np


class SampleBaselinesRunner(object):
    def __init__(self, runner, sample_rate=30):
        self.runner = runner
        self.last_batches = []
        self.sample_rate = sample_rate
        self.sample_size = 5
        self.nb_batch = 0

    def run(self):
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = self.runner.run()
        training_batch = {
            'obs': obs,
            'returns': returns,
            'actions': actions,
            'values': values,
            'neglogpacs': neglogpacs,
        }
        self.nb_batch += 1
        self.last_batches.append(training_batch)
        if len(self.last_batches) == self.sample_size:
            self.last_batches = self.last_batches[:-1]
        if self.nb_batch % self.sample_rate == self.sample_size:
            for k in ['obs', 'returns', 'actions', 'values']:
                print('----------------------------------' + k + '-------------------------------------')
                for _, batch in enumerate(self.last_batches):
                    if len(np.shape(batch[k])) == 1:
                        print(batch[k][0:10])
                    print(np.mean(batch[k]))
                    print(np.std(batch[k]))
                    print(np.shape(batch[k]))
            print('================================================')
            print(self.nb_batch)
            self.last_batches = []
        return obs, returns, masks, actions, values, neglogpacs, states, epinfos
