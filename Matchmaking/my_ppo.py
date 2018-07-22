#!/usr/bin/env python3
import gym
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gym_matchmaking
from Matchmaking.runner import EnvRunner, EnvRunnerProcess
from Matchmaking.matchmaking_agents.Policies import scripted_policy, dnn_policy, random_policy, scripted_random_policy, \
    scripted_policy_live
from Matchmaking.matchmaking_agents.Values import random_value, dnn_value
from Matchmaking.wrappers import MonitorEnv, TensorboardMathmakingEnv, TensorboardEnv
import numpy as np
from datetime import datetime
import random
import time
import multiprocessing
from docopt import docopt
from types import SimpleNamespace

_USAGE = '''
Usage:
    my_ppo (<name>)

'''
options = docopt(_USAGE)

name = str(options['<name>'])

parameters_dict = {
    "seed": 1,
    "batch_size": 128,
    "nb_epochs": 4,
    "nb_minibatch": 4,
    "clipping": 0.1,
    "learning_rate": 0.00025,
    "total_timesteps": 1000000,
}
parameters = SimpleNamespace(**parameters_dict)
parameters.total_batches = parameters.total_timesteps // parameters.batch_size
print(parameters)

date = datetime.now().strftime("%m%d-%H%M")

from Matchmaking.matchmaking_agents.models import create_model, create_atari_model


def simulate():
    results = multiprocessing.Queue()
    new_policies = multiprocessing.Queue()

    def make_session():
        import tensorflow as tf

        ncpu = multiprocessing.cpu_count()
        if sys.platform == 'darwin':
            ncpu //= 2
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=ncpu,
                                inter_op_parallelism_threads=ncpu)
        config.gpu_options.allow_growth = True  # pylint: disable=E1101
        sess = tf.Session(config=config)
        sess.__enter__()

    def make_env():
        env_name = 'CartPole-v1'
        env = gym.make(env_name)

        env.seed(parameters.seed)
        random.seed(parameters.seed)

        env = MonitorEnv(env)
        env = TensorboardEnv(env, './train/{env_name}/{name}_{date}'.format(env_name=env_name, name=name, date=date))
        return env

    def make_model(env):
        policy = dnn_policy.DNNPolicy(create_model, env, 1, 1)
        # policy_estimator = scripted_policy_live.ScriptedPolicy(env)
        value = dnn_value.DNNValue(create_model, env, 1, 1)
        return policy, value

    runner = EnvRunnerProcess(make_session, make_env, make_model, parameters, results, new_policies)
    runner.start()

    sess = make_session()
    env = make_env()
    policy_estimator, value_estimator = make_model(env)

    for t in range(parameters.total_batches):
        decay = t / parameters.total_batches
        new_policies.put(decay)

        start_time = time.time()
        training_batch, epinfos = results.get()
        print("Gather batch", time.time() - start_time)




if __name__ == "__main__":
    simulate()
