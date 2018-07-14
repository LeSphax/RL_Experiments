
import gym
import numpy as np
import random
import math
import signal
import sys
import tensorflow as tf
from time import sleep
from datetime import datetime
import time


# Initialize the "Cart-Pole" environment
env = gym.make('CartPole-v1')
# Defining the simulation related constants
EXTRA_LOG = False
NUM_EPISODES = 100000
MAX_T = 1000
SOLVED_T = 199
DEBUG_MODE = True
INPUT_SIZE = env.observation_space.shape[0]
N_HIDDEN = 64
random.seed(1224)


def fc(x, scope, nh, *, init_scale=1.0, init_bias=0.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable(
            "w", [nin, nh], initializer=tf.orthogonal_initializer(init_scale))
        b = tf.get_variable(
            "b", [nh], initializer=tf.constant_initializer(init_bias))
        return tf.matmul(x, w)+b


with tf.variable_scope("model", reuse=False):
    X = tf.placeholder(
        shape=[None, INPUT_SIZE], dtype=env.observation_space.dtype, name="X")
    activ = tf.tanh
    pi_h1 = activ(fc(X, 'pi_fc1', nh=N_HIDDEN, init_scale=np.sqrt(2)))
    pi_h2 = activ(fc(pi_h1, 'pi_fc2', nh=N_HIDDEN, init_scale=np.sqrt(2)))
    vf_h1 = activ(fc(X, 'vf_fc1', nh=N_HIDDEN, init_scale=np.sqrt(2)))
    vf_h2 = activ(fc(vf_h1, 'vf_fc2', nh=N_HIDDEN, init_scale=np.sqrt(2)))
    vf = fc(vf_h2, 'vf', 1)[:, 0]
    action_probs_out = tf.squeeze(tf.nn.softmax(activ(fc(pi_h2, 'action_probs_out', 2, init_scale=0.01, init_bias=0.0))))

DISCOUNTED_REWARD = tf.placeholder(tf.float32, [None])
ADVANTAGE = tf.placeholder(tf.float32, [None])
ACTION = tf.placeholder(tf.int32, [None])
MEAN = tf.placeholder(tf.int32, ())
MAX = tf.placeholder(tf.int32, ())
MIN = tf.placeholder(tf.int32, ())
STREAK = tf.placeholder(tf.int32, ())

tf.summary.scalar('mean', MEAN)
tf.summary.scalar('max', MAX)
tf.summary.scalar('min', MIN)
tf.summary.scalar('streak', STREAK)

action_indices = tf.range(tf.shape(ACTION)[0])
prob_indices = tf.concat([tf.reshape(action_indices, [-1, 1]), tf.reshape(ACTION, [-1, 1])], axis=1)

prob_of_picked_action = tf.gather_nd(action_probs_out, prob_indices)
policy_loss = -tf.reduce_mean(tf.log(prob_of_picked_action) * ADVANTAGE)
value_loss = tf.reduce_mean(tf.squared_difference(vf, DISCOUNTED_REWARD))
loss = value_loss + policy_loss
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss, global_step=tf.train.get_global_step())

merged = tf.summary.merge_all()
sess = tf.Session()

train_writer = tf.summary.FileWriter(
    './train/{date}'.format(date=datetime.now().strftime("%Y%m%d-%H%M%S")), sess.graph)


def reset_episodes():
    return {
        'timesteps': [],
        'actions': [],
        'obs': [],
        'discounted_rewards': [],
        'advantages': [],
    }


def simulate():

    # Instantiating the learning related parameters

    num_streaks = 0
    highest_streak = 0

    init = tf.global_variables_initializer()
    sess.run(init)
    episodes = reset_episodes()
    episodes_for_logging = reset_episodes()
    start = 0

    timers = {
        'action': 0,
        'value': 0,
        'step': 0,
        'stuff': 0,
        'train': 0,
        'log': 0,
    }
    for episode in range(NUM_EPISODES):
        if(episode > 0 and episode % 100 == 0):
            print("Action %s" % timers['action'])
            print("Value %s" % timers['value'])
            print("Step %s" % timers['step'])
            print("Stuff %s" % timers['stuff'])
            print("Train %s" % timers['train'])
            print("Log %s" % timers['log'])
        experience = {
            'actions': [],
            'values': [],
            'obs': [],
            'rewards': [],
            'dones': [],
        }

        # Reset the environment

        obs = env.reset()
        for t in range(MAX_T):
            # env.render()
            # Execute the action
            x = np.reshape(obs, [1, INPUT_SIZE])
            start = time.time()
            action_probs = sess.run(action_probs_out, {X: x})
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            timers['action'] += time.time() - start
            start = time.time()
            value = sess.run(vf, {X: x})            
            value = np.asscalar(value)
            timers['value'] += time.time() - start
            start = time.time()
            obs, reward, done, _ = env.step(action)
            timers['step'] += time.time() - start
            start = time.time()
            experience['actions'].append(action)
            experience['values'].append(value)
            experience['obs'].append(obs)
            experience['rewards'].append(reward)
            experience['dones'].append(done)

            if t == MAX_T-1:
                done = True

            if done:
                if (t >= SOLVED_T):
                    num_streaks += 1
                else:
                    num_streaks = 0
                break

        discounted_rewards = []
        advantages = []
        previous_reward = 0
        discount_factor = 0.99

        experience['values'].append(0)
        for idx, reward in reversed(list(enumerate(experience['rewards']))):
            previous_reward = discount_factor * previous_reward + reward
            discounted_rewards.insert(0, previous_reward)
            advantages.insert(0, experience['values'][idx+1] + reward - experience['values'][idx])

        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-8)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        episodes['timesteps'].append(t)
        episodes['actions'].append(experience['actions'])
        episodes['obs'].append(experience['obs'])
        episodes['discounted_rewards'].append(discounted_rewards)
        episodes['advantages'].append(advantages)
        highest_streak = max(highest_streak, num_streaks)
        timers['stuff'] += time.time() - start
        if(True or episode > 0 and episode % 5 == 0):
            start = time.time()
            for idx in range(len(episodes['timesteps'])):
                vl, pl, _ = sess.run(
                    [value_loss, policy_loss, train],
                    {
                        X: episodes['obs'][idx],
                        DISCOUNTED_REWARD: episodes['discounted_rewards'][idx],
                        ADVANTAGE: episodes['advantages'][idx],
                        ACTION: episodes['actions'][idx]
                    }
                )
            timers['train'] += time.time() - start
            start = time.time()
            episodes_for_logging['timesteps'] = np.concatenate((episodes_for_logging['timesteps'], episodes['timesteps']))
            if(episode > 0 and episode % 100 == 0):
                mean_timesteps = np.mean(episodes_for_logging['timesteps'])
                max_timesteps = np.max(episodes_for_logging['timesteps'])
                min_timesteps = np.min(episodes_for_logging['timesteps'])
                summary = sess.run(
                    merged,
                    {
                        MEAN: mean_timesteps,
                        MAX: max_timesteps,
                        MIN: min_timesteps,
                        STREAK: highest_streak
                    }
                )
                train_writer.add_summary(summary, episode)
                print("Episode %s Reward %s Max %s Min %s Streak %s" % (episode, mean_timesteps, max_timesteps, min_timesteps, highest_streak))
                highest_streak = 0
                episodes_for_logging = reset_episodes()
            timers['log'] += time.time() - start
            episodes = reset_episodes()

        if EXTRA_LOG:
            pl, ai, pi, apo, ppa = sess.run(
                [policy_loss, action_indices, prob_indices, action_probs_out, prob_of_picked_action],
                {
                    X: experience['obs'],
                    ADVANTAGE: discounted_rewards,
                    ACTION: experience['actions'],
                }
            )
            print("Timesteps %s" % t)
            print("Computed %s %s %s %s %s" % (pl, ai, pi, apo, ppa))
            # print("Values %s %s" % (np.round(experience['values'], 1)[0:10], np.round(experience['values'], 1)[-10:]))
            print("Rewards %s %s %s" % (np.round(discounted_rewards, 1)[0:5], np.round(discounted_rewards, 1)[-5:], len(discounted_rewards)))
            print("Actions %s %s %s" % (np.round(experience['actions'], 1)[0:5], np.round(experience['actions'], 1)[-5:], len(experience['actions'])))
            # print("Adv %s %s" % (np.round(advantages, 1)[0:3], np.round(advantages, 1)[-3:]))


if __name__ == "__main__":
    simulate()
