#!/usr/bin/env python3
import gym
import sys
import os
import _thread


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Matchmaking.atari import StateProcessor, create_model, ProcessStateEnv
from Matchmaking.runner import EnvRunner
from Matchmaking.matchmaking_agents.Policies import dnn_policy
from Matchmaking.matchmaking_agents.Values import dnn_value
from Matchmaking.wrappers import MonitorEnv, TensorboardEnv, AutoResetEnv

import numpy as np
import utils.keyPoller as kp
from datetime import datetime
import random
import time
import multiprocessing
from docopt import docopt
from types import SimpleNamespace
import tensorflow as tf

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
    "learning_rate": 0.000025,
    "total_timesteps": 1000000,
}
parameters = SimpleNamespace(**parameters_dict)
parameters.total_batches = parameters.total_timesteps // parameters.batch_size
print(parameters)

date = datetime.now().strftime("%m%d-%H%M")



# def create_summaries(save_path):


def simulate():
    env_name = 'Breakout-v0'
    save_path = './train/{env_name}/{name}_{date}'.format(env_name=env_name, name=name, date=date)

    def make_session():
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
        env = gym.make(env_name)

        env.seed(parameters.seed)
        random.seed(parameters.seed)

        env = MonitorEnv(env)
        env = ProcessStateEnv(env)
        return env

    def make_model(env):
        policy = dnn_policy.DNNPolicy(create_model, env, 1)
        # policy_estimator = scripted_policy_live.ScriptedPolicy(env)
        value = dnn_value.DNNValue(create_model, env, 1)
        return policy, value

    make_session()
    env = make_env()
    policy_estimator, value_estimator = make_model(env)
    saver = tf.train.Saver()
    env = TensorboardEnv(env, saver, save_path)

    def renderer_thread(make_env, policy_estimator):
        env = make_env()
        env = AutoResetEnv(env)
        obs = env.reset()
        render = False

        def toggle_rendering():
            print("Toggle rendering")
            nonlocal render
            render = not render

        while True:
            kp.keyboardCommands("r", toggle_rendering)
            if render:
                env.render()
                action, neglogp_action = policy_estimator.get_action(obs)

                obs, reward, done, info = env.step(action)

    sess = tf.get_default_session()

    _thread.start_new_thread(renderer_thread, (make_env, policy_estimator))

    runner = EnvRunner(sess, env, policy_estimator, value_estimator)
    for t in range(parameters.total_batches):

        decay = t / parameters.total_batches
        learning_rate = parameters.learning_rate * (1 - decay)
        clipping = parameters.clipping * (1 - decay)

        start_time = time.time()
        training_batch, epinfos = runner.run_timesteps(parameters.batch_size)
        print("Run batch", time.time() - start_time)

        start_time = time.time()
        inds = np.arange(parameters.batch_size)
        for _ in range(parameters.nb_epochs):

            np.random.shuffle(inds)
            minibatch_size = parameters.batch_size // parameters.nb_minibatch
            for start in range(0, parameters.batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = inds[start:end]

                entropy, policy_loss, policy_gradient_step = policy_estimator.get_gradients(
                    obs=training_batch['obs'][mb_inds],
                    actions=training_batch['actions'][mb_inds],
                    neglogp_actions=training_batch['neglogp_actions'][mb_inds],
                    advantages=training_batch['advantages'][mb_inds],
                    clipping=clipping,
                    learning_rate=learning_rate,
                )
                policy_estimator.apply_gradients(policy_gradient_step, learning_rate)

                value_loss, value_gradient_step = value_estimator.get_gradients(
                    obs=training_batch['obs'][mb_inds],
                    values=training_batch['values'][mb_inds],
                    returns=training_batch['returns'][mb_inds],
                    clipping=clipping,
                    learning_rate=learning_rate,
                )
                value_estimator.apply_gradients(value_gradient_step, learning_rate)

        print("Train model", time.time() - start_time)


if __name__ == "__main__":
    simulate()
