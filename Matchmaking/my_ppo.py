#!/usr/bin/env python3
import gym
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gym_matchmaking
from Matchmaking.matchmaking_agents.Policies import scripted_policy, dnn_policy, random_policy, scripted_random_policy, \
    scripted_policy_live
from Matchmaking.matchmaking_agents.Values import random_value, dnn_value
from Matchmaking.wrappers import AutoResetEnv, MonitorEnv, TensorboardMathmakingEnv, TensorboardEnv, NormalizeEnv
import numpy as np
import tensorflow as tf
from datetime import datetime
import utils.keyPoller as kp
import random
import time
import multiprocessing
from docopt import docopt

_USAGE = '''
Usage:
    my_ppo (<name>)

'''
options = docopt(_USAGE)

name = str(options['<name>'])

SEED = 1

SUMMARY_INTERVAL = 10
BATCH_SIZE = 128
NB_EPOCHS = 4
NB_MINIBATCH = 4
CLIPPING = 0.1
LEARNING_RATE = 0.00025
TOTAL_TIMESTEPS = 1000000
TOTAL_BATCHES = TOTAL_TIMESTEPS // BATCH_SIZE

date = datetime.now().strftime("%m%d-%H%M")


class EnvRunner(object):
    def __init__(self, env, value_estimator, policy_estimator, discount_factor=0.9999, gae_weighting=0.95):
        self.env = AutoResetEnv(env, 300)
        if policy_estimator.normalize():
            self.env = NormalizeEnv(self.env)
        self.obs = self.env.reset()
        self.value_estimator = value_estimator
        self.policy_estimator = policy_estimator
        self.discount_factor = discount_factor
        self.gae_weighting = gae_weighting

    def run_timesteps(self, nb_timesteps):
        batch = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'dones': [],
            'neglogp_actions': [],
            'advantages': [],
            'returns': []
        }
        epinfos = []

        for t in range(nb_timesteps):
            kp.checkKeyStrokes(None, name)
            if kp.render:
                self.env.render()

            batch['obs'].append(self.obs)
            value = self.value_estimator.get_value(self.obs)
            batch['values'].append(value)

            action, neglogp_action = self.policy_estimator.get_action(self.obs)
            self.obs, reward, done, info = self.env.step(action)

            batch['neglogp_actions'].append(neglogp_action)
            batch['actions'].append(action)
            batch['rewards'].append(reward)
            batch['dones'].append(done)

            maybeepinfo = info.get('episode')
            if maybeepinfo:
                epinfos.append(maybeepinfo)

        advantages = np.zeros_like(batch['rewards'], dtype=float)

        next_value = 0 if done else self.value_estimator.get_value(self.obs)
        batch['values'] = np.append(batch['values'], next_value)
        last_discounted_adv = 0
        for idx in reversed(range(nb_timesteps)):
            if batch['dones'][idx] == 1:
                next_value = 0
                use_last_discounted_adv = 0
            else:
                next_value = batch['values'][idx + 1]
                use_last_discounted_adv = 1

            td_error = self.discount_factor * next_value + batch['rewards'][idx] - batch['values'][idx]
            advantages[
                idx] = last_discounted_adv = td_error + self.discount_factor * self.gae_weighting * last_discounted_adv * use_last_discounted_adv
        returns = advantages + batch['values'][:-1]

        batch['advantages'] = advantages
        batch['returns'] = returns

        return {k: np.asarray(batch[k]) for k in batch}, epinfos


def simulate():
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin':
        ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True  # pylint: disable=E1101
    sess = tf.Session(config=config)
    sess.__enter__()

    env_name = 'Breakout-v0'
    env = gym.make(env_name)

    env.seed(SEED)
    random.seed(SEED)

    env = MonitorEnv(env)
    env = TensorboardEnv(env, './train/{env_name}/{name}_{date}'.format(env_name=env_name, name=name, date=date))

    policy_estimator = dnn_policy.DNNPolicy(env, 1, 1)
    # policy_estimator = scripted_policy_live.ScriptedPolicy(env)
    value_estimator = dnn_value.DNNValue(env, 1, 1)

    runner = EnvRunner(env, value_estimator, policy_estimator)

    for t in range(TOTAL_BATCHES):
        decay = t / TOTAL_BATCHES

        start_time = time.time()
        training_batch, epinfos = runner.run_timesteps(BATCH_SIZE)
        print("Gather batch", time.time() - start_time)

        start_time = time.time()
        inds = np.arange(BATCH_SIZE)
        for _ in range(NB_EPOCHS):

            np.random.shuffle(inds)
            minibatch_size = BATCH_SIZE // NB_MINIBATCH
            for start in range(0, BATCH_SIZE, minibatch_size):
                end = start + minibatch_size
                mb_inds = inds[start:end]

                entropy, policy_loss = policy_estimator.train_model(
                    obs=training_batch['obs'][mb_inds],
                    actions=training_batch['actions'][mb_inds],
                    neglogp_actions=training_batch['neglogp_actions'][mb_inds],
                    advantages=training_batch['advantages'][mb_inds],
                    clipping=CLIPPING * (1 - decay),
                    learning_rate=LEARNING_RATE * (1 - decay),
                )

                value_loss = value_estimator.train_model(
                    obs=training_batch['obs'][mb_inds],
                    values=training_batch['values'][mb_inds],
                    returns=training_batch['returns'][mb_inds],
                    clipping=CLIPPING * (1 - decay),
                    learning_rate=LEARNING_RATE * (1 - decay),
                )
        print("Train model", time.time() - start_time)


if __name__ == "__main__":
    simulate()
