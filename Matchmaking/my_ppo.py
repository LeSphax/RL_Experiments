#!/usr/bin/env python3
import _thread
import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Matchmaking.wrappers.sample_runner import SampleRunner
from Matchmaking.breakout_no_frameskip_config import BreakoutNoFrameskipConfig

from Matchmaking.cartpole_config import CartPoleConfig
from Matchmaking.runner import EnvRunner
from Matchmaking.matchmaking_agents.Policies import dnn_policy
from Matchmaking.matchmaking_agents.Values import dnn_value
from Matchmaking.atari_config import AtariConfig

import numpy as np
import utils.keyPoller as kp
from datetime import datetime
import time
import multiprocessing
from docopt import docopt
import tensorflow as tf

_USAGE = '''
Usage:
    my_ppo (<name>) [--debug]
    
Options:
    --debug                        Tensorflow debugger

'''
options = docopt(_USAGE)

name = str(options['<name>'])

date = datetime.now().strftime("%m%d-%H%M")


def simulate():
    config = BreakoutNoFrameskipConfig()
    parameters = config.parameters
    save_path = './train/{env_name}/{name}_{date}'.format(env_name=config.env_name, name=name, date=date)

    def make_session():
        ncpu = multiprocessing.cpu_count()
        if sys.platform == 'darwin':
            ncpu //= 2
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=ncpu,
                                inter_op_parallelism_threads=ncpu)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.__enter__()

    def make_model(env):
        policy = dnn_policy.DNNPolicy(config.create_model, env, reuse_model=False)
        value = dnn_value.DNNValue(config.create_model, env, reuse_model=True)
        return policy, value

    make_session()
    sess = tf.get_default_session()

    venv = config.make_vec_env(save_path)
    policy_estimator, value_estimator = make_model(venv)
    saver = tf.train.Saver()

    def renderer_thread(policy_estimator, sess):
        with sess.as_default(), sess.graph.as_default():
            env = config.make_vec_env(renderer=True)
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
                    time.sleep(0.02)
                else:
                    time.sleep(1)

    _thread.start_new_thread(renderer_thread, (policy_estimator, sess))

    runner = EnvRunner(sess, venv, policy_estimator, value_estimator)
    # runner = SampleRunner(runner)
    for t in range(parameters.total_batches):

        decay = t / parameters.total_batches if parameters.decay else 0
        learning_rate = parameters.learning_rate * (1 - decay)
        clipping = parameters.clipping * (1 - decay)

        training_batch, epinfos = runner.run_timesteps(parameters.batch_size)

        inds = np.arange(parameters.batch_size)
        for _ in range(parameters.nb_epochs):

            np.random.shuffle(inds)
            minibatch_size = parameters.batch_size // parameters.nb_minibatch
            for start in range(0, parameters.batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = inds[start:end]

                entropy, policy_loss = policy_estimator.train(
                    obs=training_batch['obs'][mb_inds],
                    actions=training_batch['actions'][mb_inds],
                    neglogp_actions=training_batch['neglogp_actions'][mb_inds],
                    advantages=training_batch['advantages'][mb_inds],
                    clipping=clipping,
                    learning_rate=learning_rate,
                )

                value_loss = value_estimator.train(
                    obs=training_batch['obs'][mb_inds],
                    values=training_batch['values'][mb_inds],
                    returns=training_batch['returns'][mb_inds],
                    clipping=clipping,
                    learning_rate=learning_rate,
                )

        if t % (1000) == 0:
            print("Saved model", t)
            saver.save(sess, save_path + "/latest_save")


if __name__ == "__main__":
    simulate()
