import gym
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gym_matchmaking
import matchmaking_agents
from matchmaking_agents.random_agent import RandomAgent
from matchmaking_agents.dnn_agent import DNNAgent
from matchmaking_agents.dnn_agent_scripted_policy import DNNScriptedAgent
from matchmaking_agents.scripted_agent import ScriptedAgent
from matchmaking_agents.reward_per_timestep import RewardPerTimestep
import numpy as np
import tensorflow as tf
from datetime import datetime
from utils.keyPoller import checkKeyStrokes
import random

env = gym.make('Matchmaking-v0')
random.seed(1)
env.seed(1)

name = datetime.now().strftime("%Y%m%d-%H%M%S")


def simulate():

    
    # agent = DNNAgent(env)
    agent = DNNScriptedAgent(env)
    # agent = RandomAgent(env)
    # agent = ScriptedAgent(env)
    MEAN = tf.placeholder(tf.float32, ())
    TOTAL_REWARD = tf.placeholder(tf.float32, ())
    VALUE = tf.placeholder(tf.float32, [None])
    DISCOUNTED_REWARDS = tf.placeholder(tf.float32, ())
    ADVANTAGES = tf.placeholder(tf.float32, ())
    VALUE_LOSS = tf.placeholder(tf.float32, ())
    POLICY_LOSS = tf.placeholder(tf.float32, ())

    tf.summary.scalar('mean', MEAN)
    tf.summary.scalar('total_reward', TOTAL_REWARD)
    tf.summary.histogram('values', VALUE)
    tf.summary.scalar('discounted_rewards', DISCOUNTED_REWARDS)
    tf.summary.scalar('advantages', ADVANTAGES)
    tf.summary.scalar('value_loss', VALUE_LOSS)
    tf.summary.scalar('policy_loss', POLICY_LOSS)
    merged = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter('./train/{name}'.format(name=NAME), sess.graph)
    render = False    
    for e in range(100000):
        experiences = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'values': [],
        }
        obs = env.reset()
        for t in range(20):
            checkKeyStrokes(sess,name)
            if render:
                env.render()
            chosen_players, value = agent.get_action_and_value(obs)
            experiences['obs'].append(obs)
            obs, reward, done, _ = env.step(chosen_players)
            average_reward.add_reward(reward)
            experiences['actions'].append(chosen_players)
            experiences['rewards'].append(reward)
            experiences['values'].append(value)
            # print(np.reshape(experiences['values'], [-1])[0:10])
        # print(np.mean(experiences['rewards']))
        discounted_rewards, advantages, returns, value_loss, policy_loss = agent.train_model(experiences['obs'], experiences['actions'], experiences['rewards'], experiences['values'])
        summary = sess.run(
            merged,
            {
                MEAN: average_reward.average,
                TOTAL_REWARD: np.sum(experiences['rewards']),
                VALUE: np.reshape(experiences['values'], [-1]),
                DISCOUNTED_REWARDS: np.mean(discounted_rewards),
                ADVANTAGES: np.mean(advantages),
                VALUE_LOSS: value_loss,
                POLICY_LOSS: policy_loss
            }
        )
        print(value_loss)
        # print(policy_loss)
        # print("Values ", np.reshape(experiences['values'],[-1]))
        # print("Rewards ", experiences['rewards'])
        # print("Discounted_rewards ", np.round(np.reshape(discounted_rewards, [-1]),1)[0:100])
        # print("Advantages ", np.round(np.reshape(advantages, [-1]),1)[0:100])
        # print("Returns ", np.round(np.reshape(returns, [-1]),1)[0:100])
        train_writer.add_summary(summary, e)


if __name__ == "__main__":
    simulate()
