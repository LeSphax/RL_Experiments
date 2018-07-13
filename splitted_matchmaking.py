import gym
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gym_matchmaking
import matchmaking_agents
from matchmaking_agents.Policies import scripted_policy, dnn_policy, random_policy, scripted_random_policy
from matchmaking_agents.Values import random_value, dnn_value
import numpy as np
import tensorflow as tf
from datetime import datetime
from utils.gae import gae
import utils.keyPoller as kp
import random

env = gym.make('Matchmaking-v0')
random.seed(12)
env.seed(12)

name = datetime.now().strftime("%Y%m%d-%H%M%S")


def simulate():

    policy_estimator = dnn_policy.DNNPolicy(env,1)
    value_estimator = dnn_value.DNNValue(env,0)

    INCORRECT_ACTIONS = tf.placeholder(tf.float32, ())
    HOLDS_NOT_EMPTY = tf.placeholder(tf.float32, ())
    HOLDS = tf.placeholder(tf.float32, ())
    TOTAL_REWARD = tf.placeholder(tf.float32, ())
    VALUE = tf.placeholder(tf.float32, [None])
    ADVANTAGES = tf.placeholder(tf.float32, [None])
    VALUE_LOSS = tf.placeholder(tf.float32, ())
    POLICY_LOSS = tf.placeholder(tf.float32, ())

    tf.summary.scalar('incorrect_actions', INCORRECT_ACTIONS)
    tf.summary.scalar('holds_not_empty', HOLDS_NOT_EMPTY)
    tf.summary.scalar('holds', HOLDS)
    tf.summary.scalar('total_reward', TOTAL_REWARD)
    tf.summary.histogram('values', VALUE)
    tf.summary.histogram('advantages', ADVANTAGES)
    tf.summary.scalar('value_loss', VALUE_LOSS)
    tf.summary.scalar('policy_loss', POLICY_LOSS)
    merged = tf.summary.merge_all()
    global_step = tf.Variable(0, name="global_step", trainable=False)
    sess = tf.Session()
    train_writer = tf.summary.FileWriter('./train/{name}'.format(name=name), sess.graph)

    for e in range(1000000):
        experiences = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'values': [],
        }
        obs = env.reset()
        for t in range(20):
            kp.checkKeyStrokes(sess, name)
            if kp.render:
                env.render()

            experiences['obs'].append(obs)
            value = value_estimator.get_value(obs)
            experiences['values'].append(value)
            
            chosen_players = policy_estimator.get_action(obs)
            obs, reward, done, _ = env.step(chosen_players)

            experiences['actions'].append(chosen_players)
            experiences['rewards'].append(reward)

        next_value = 0 if done else value_estimator.get_value(obs)

        advantages, returns = gae(
            rewards=experiences['rewards'],
            values=experiences['values'],
            next_value=next_value
        )

        policy_loss = policy_estimator.train_model(
            obs=experiences['obs'],
            actions=experiences['actions'],
            advantages=advantages
        )

        value_loss = value_estimator.train_model(
            obs=experiences['obs'],
            returns=returns,
        )

       

        actions = np.array(experiences['actions'])
        rewards = np.array(experiences['rewards'])
        observations = np.array(experiences['obs'])

        non_empty_obs = np.logical_or.reduce(observations != -1, axis =1)
        holds = np.logical_and.reduce(actions == [10,10], axis =1)
        holds_not_empty = np.logical_and(holds, non_empty_obs)

        summary = sess.run(
            merged,
            {
                INCORRECT_ACTIONS: rewards[np.where(rewards == -0.1)].size,
                HOLDS_NOT_EMPTY: holds_not_empty[holds_not_empty].size,
                HOLDS: holds[holds].size,
                TOTAL_REWARD: np.sum(experiences['rewards']),
                VALUE: np.reshape(experiences['values'], [-1]),
                ADVANTAGES: advantages,
                VALUE_LOSS: value_loss,
                POLICY_LOSS: policy_loss
            }
        )
        print("Actions ", np.reshape(experiences['actions'],[-1]))
        print("Values ", np.reshape(experiences['values'],[-1]))
        print("Rewards ", experiences['rewards'])
        print("Advantages ", np.round(np.reshape(advantages, [-1]),1)[0:100])
        print("Returns ", np.round(np.reshape(returns, [-1]),1)[0:100])
        train_writer.add_summary(summary, e)


if __name__ == "__main__":
    simulate()