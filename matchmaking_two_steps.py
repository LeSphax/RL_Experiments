import gym
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gym_matchmaking
import matchmaking_agents
from matchmaking_agents.Policies.Two_steps import scripted_policy, dnn_policy, random_policy, scripted_random_policy
from matchmaking_agents.Values import random_value, dnn_value
import numpy as np
import tensorflow as tf
from datetime import datetime
from utils.gae import gae
import utils.keyPoller as kp
import random
import time

env = gym.make('CartPole-v1')
seed = 1
random.seed(seed)
env.seed(seed)
SUMMARY_INTERVAL = 100

name = datetime.now().strftime("%Y%m%d-%H%M%S")


def run_episode(value_estimator, policy_estimator):
    trajectory = {
        'obs': [],
        'actions': [],
        'rewards': [],
        'values': [],
        'neglogp_actions': []
    }
    obs = env.reset()
    for t in range(500):
        kp.checkKeyStrokes(None, name)
        if kp.render:
            env.render()

        trajectory['obs'].append(obs)
        value = value_estimator.get_value(obs)
        trajectory['values'].append(value)

        action, neglogp_action = policy_estimator.get_action(obs)
        obs, reward, done, _ = env.step(action)

        trajectory['neglogp_actions'].append(neglogp_action)
        trajectory['actions'].append(action)
        trajectory['rewards'].append(reward)
        if done:
            break

    next_value = 0 if done else value_estimator.get_value(obs)

    advantages, returns = gae(
        rewards=trajectory['rewards'],
        values=trajectory['values'],
        next_value=next_value,
        discount_factor=0.95
    )

    return {
        'obs': trajectory['obs'],
        'actions': trajectory['actions'],
        'neglogp_actions': trajectory['neglogp_actions'],
        'rewards': trajectory['rewards'],
        'values': trajectory['values'],
        'next_value': [next_value],
        'advantages': advantages,
        'returns': returns
    }


def simulate():

    policy_estimator = dnn_policy.DNNPolicy(env, 2)
    value_estimator = dnn_value.DNNValue(env, 2)

    ENTROPY = tf.placeholder(tf.float32, ())
    FPS = tf.placeholder(tf.float32, ())
    # INCORRECT_ACTIONS = tf.placeholder(tf.float32, ())
    # HOLDS_NOT_EMPTY = tf.placeholder(tf.float32, ())
    # HOLDS = tf.placeholder(tf.float32, ())
    VALUE_LOSS = tf.placeholder(tf.float32, ())
    POLICY_LOSS = tf.placeholder(tf.float32, ())
    TOTAL_REWARD = tf.placeholder(tf.float32, ())
    TOTAL_TIMESTEPS = tf.placeholder(tf.float32, ())

    tf.summary.scalar('entropy', ENTROPY)
    tf.summary.scalar('fps', FPS)
    tf.summary.scalar('total_timesteps', TOTAL_TIMESTEPS)
    # tf.summary.scalar('incorrect_actions', INCORRECT_ACTIONS)
    # tf.summary.scalar('holds_not_empty', HOLDS_NOT_EMPTY)
    # tf.summary.scalar('holds', HOLDS)
    tf.summary.scalar('value_loss', VALUE_LOSS)
    tf.summary.scalar('policy_loss', POLICY_LOSS)
    tf.summary.scalar('total_reward', TOTAL_REWARD)

    merged = tf.summary.merge_all()
    global_step = tf.Variable(0, name="global_step", trainable=False)
    sess = tf.Session()
    train_writer = tf.summary.FileWriter('./train/{name}'.format(name=name), sess.graph)

    training_batch = []
    summary_batch = {
        'rewards': [],
        'value_losses': [],
        'policy_losses': [],
        'entropies': []
    }
    total_timesteps = 0
    start = time.time()

    for e in range(1000000):
        episode = run_episode(value_estimator, policy_estimator)

        training_batch.append(episode)

        if len(training_batch) == 10:
            flat_training_batch = {k: np.concatenate([np.array(dic[k]) for dic in training_batch]) for k in training_batch[0]}

            entropy, policy_loss = policy_estimator.train_model(
                obs=flat_training_batch['obs'],
                actions=flat_training_batch['actions'],
                neglogp_actions=flat_training_batch['neglogp_actions'],
                advantages=flat_training_batch['advantages'],
            )

            value_loss = value_estimator.train_model(
                obs=flat_training_batch['obs'],
                returns=flat_training_batch['returns'],
            )

            summary_batch['rewards'].append(flat_training_batch['rewards'])
            summary_batch['value_losses'].append([value_loss])
            summary_batch['policy_losses'].append([policy_loss])
            summary_batch['entropies'].append([entropy])
            training_batch = []

        if e % SUMMARY_INTERVAL == 0 and e != 0:
            flat_summary_batch = {k: np.concatenate([np.array(list) for list in summary_batch[k]]) for k in summary_batch}

            # actions = np.array(trajectory['actions'])
            # rewards = np.array(trajectory['rewards'])
            # observations = np.array(trajectory['obs'])

            # non_empty_obs = np.logical_or.reduce(observations != -1, axis =1)
            # holds = actions == 10
            # holds_not_empty = np.logical_and(holds, non_empty_obs)

            print("Timesteps ", len(episode['values']))
            print("Values ", episode['values'], episode['next_value'])
            print("Rewards ", episode['rewards'])
            print("Advantages ", episode['advantages'])
            print("Returns ", episode['returns'])

            timesteps = len(flat_summary_batch['rewards'])
            total_timesteps += timesteps
            time_between_summaries = time.time() - start
            start = time.time()
            summary = sess.run(
                merged,
                {
                    ENTROPY:  np.mean(flat_summary_batch['entropies']),
                    FPS: timesteps/time_between_summaries,
                    TOTAL_TIMESTEPS: total_timesteps,
                    # INCORRECT_ACTIONS: rewards[np.where(rewards == -0.1)].size,
                    # HOLDS_NOT_EMPTY: holds_not_empty[holds_not_empty].size,
                    # HOLDS: holds[holds].size,
                    VALUE_LOSS: np.mean(flat_summary_batch['value_losses']),
                    POLICY_LOSS: np.mean(flat_summary_batch['policy_losses']),
                    TOTAL_REWARD: np.sum(flat_summary_batch['rewards']) / SUMMARY_INTERVAL,
                }
            )
            summary_batch = {
                'rewards': [],
                'value_losses': [],
                'policy_losses': [],
                'entropies': []
            }
            train_writer.add_summary(summary, e)


if __name__ == "__main__":
    simulate()
