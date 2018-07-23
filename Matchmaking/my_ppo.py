#!/usr/bin/env python3
from multiprocessing import Pipe

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
from Matchmaking.renderer import EnvRendererProcess

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

from Matchmaking.matchmaking_agents.models import create_model, create_atari_model


# def create_summaries(save_path):


def simulate():
    env_name = 'Breakout-v0'
    save_path = './train/{env_name}/{name}_{date}'.format(env_name=env_name, name=name, date=date)

    results = multiprocessing.Queue()
    nb_workers = 4
    new_policy_queues = [None for _ in range(nb_workers)]
    print(new_policy_queues)

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

        # env.seed(parameters.seed)
        random.seed(parameters.seed)

        env = MonitorEnv(env)
        # env = TensorboardEnv(env, save_path)
        return env

    def make_model(env):
        policy = dnn_policy.DNNPolicy(create_atari_model, env, 1,1)
        # policy_estimator = scripted_policy_live.ScriptedPolicy(env)
        value = dnn_value.DNNValue(create_atari_model, env, 1, 1)
        return policy, value

    for i in range(nb_workers):
        new_policy_queues[i] = multiprocessing.Queue()
        runner = EnvRunnerProcess(i, make_session, make_env, make_model, parameters, results, new_policy_queues[i])
        runner.start()

    renderer_new_policy_queue = multiprocessing.Queue()
    instruction_pipe, child_pipe = Pipe()
    renderer = EnvRendererProcess(make_session, make_env, make_model, renderer_new_policy_queue, child_pipe)
    renderer.start()

    make_session()
    env = make_env()
    policy_estimator, value_estimator = make_model(env)

    render = False

    def toggleRendering():
        print("Toggle rendering")
        nonlocal render
        if not render:
            render = True
            policy_weights = policy_estimator.get_weights()
            value_weights = value_estimator.get_weights()

            renderer_new_policy_queue.put((policy_weights, value_weights))
            instruction_pipe.send('start')
        else:
            render = False
            instruction_pipe.send('stop')

    TOTAL_REWARD = tf.placeholder(tf.float32, ())
    FPS = tf.placeholder(tf.float32, ())
    tf.summary.scalar('total_reward', TOTAL_REWARD)
    tf.summary.scalar('fps', FPS)
    merged = tf.summary.merge_all()
    sess = tf.get_default_session()
    train_writer = tf.summary.FileWriter(save_path, sess.graph)

    eprewards = []
    previous_summary_time = time.time()
    for t in range(parameters.total_batches):
        kp.keyboardCommands("r", toggleRendering)

        decay = t / parameters.total_batches
        learning_rate = parameters.learning_rate * (1 - decay)
        start_time = time.time()
        policy_weights = policy_estimator.get_weights()
        value_weights = value_estimator.get_weights()

        for i in range(nb_workers):
            new_policy_queues[i].put((decay, policy_weights, value_weights))
        # print("Get and write weights", time.time() - start_time)

        start_time = time.time()
        gradients_policy, gradients_value = [], []
        for i in range(nb_workers):
            worker_p_grads, worker_v_grads, epinfos = results.get()
            eprewards = np.concatenate([eprewards, [epinfo['total_reward'] for epinfo in epinfos]])
            gradients_policy.append(worker_p_grads)
            gradients_value.append(worker_v_grads)

        policy_estimator.apply_gradients(np.mean(gradients_policy, axis=0), learning_rate * parameters.nb_minibatch * parameters.nb_epochs)
        value_estimator.apply_gradients(np.mean(gradients_value, axis=0), learning_rate * parameters.nb_minibatch * parameters.nb_epochs)

        if t % 10 == 0 and t > 0:
            print(t, "Run summary", np.mean(eprewards), 10 * parameters.batch_size, t * parameters.batch_size)

            summary = sess.run(merged, {
                TOTAL_REWARD: np.mean(eprewards),
                FPS: 10 * parameters.batch_size / (time.time() - previous_summary_time)
            })
            train_writer.add_summary(summary, t * parameters.batch_size)

            previous_summary_time = time.time()
            eprewards=[]
        # print("Gather batch", time.time() - start_time)


if __name__ == "__main__":
    simulate()
