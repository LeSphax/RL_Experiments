
import gym
import numpy as np
import random
import math
import signal
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorflow as tf
from time import sleep
from datetime import datetime
import utils.gae
import time
import utils.keyPoller as kp


# Initialize the "Cart-Pole" environment
env = gym.make('CartPole-v1')
# Defining the simulation related constants
EXTRA_LOG = False
NUM_EPISODES = 100000
MAX_T = 10000000
SOLVED_T = 199
DEBUG_MODE = True
N_HIDDEN = 32
NAME = datetime.now().strftime("%Y%m%d-%H%M%S")
random.seed(1224)
OUT_SIZE = env.action_space.n


def fc(x, scope, nh, *, init_scale=1.0, init_bias=0.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable(
            "w", [nin, nh], initializer=tf.orthogonal_initializer(init_scale))
        b = tf.get_variable(
            "b", [nh], initializer=tf.constant_initializer(init_bias))
        return tf.matmul(x, w)+b


with tf.variable_scope("model", reuse=False):
    X = tf.placeholder(shape=[None, env.observation_space.shape[0]], dtype=env.observation_space.dtype, name="X")
    activ = tf.tanh
    pi_h1 = activ(fc(X, 'pi_fc1', nh=N_HIDDEN, init_scale=np.sqrt(2)))
    pi_h2 = activ(fc(pi_h1, 'pi_fc2', nh=N_HIDDEN, init_scale=np.sqrt(2)))
    vf_h1 = activ(fc(X, 'vf_fc1', nh=N_HIDDEN, init_scale=np.sqrt(2)))
    vf_h2 = activ(fc(vf_h1, 'vf_fc2', nh=N_HIDDEN, init_scale=np.sqrt(2)))
    vf = fc(vf_h2, 'vf', 1)[:, 0]
    action_probs_out = tf.squeeze(tf.nn.softmax(activ(fc(pi_h2, 'action_probs_out', OUT_SIZE, init_scale=0.01, init_bias=0.0))))

RETURNS = tf.placeholder(tf.float32, [None])
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
value_loss = tf.reduce_mean(tf.squared_difference(vf, RETURNS))
loss = value_loss + policy_loss
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss, global_step=tf.train.get_global_step())

merged = tf.summary.merge_all()
sess = tf.Session()

train_writer = tf.summary.FileWriter('./train/{name}'.format(name=NAME), sess.graph)

def reset_episodes():
    return {
        'timesteps': [],
        'actions': [],
        'obs': [],
        'returns': [],
        'advantages': [],
    }

def run_episode(render):
    experience = {
        'actions': [],
        'values': [],
        'obs': [],
        'rewards': [],
        'dones': [],
    }
    obs = env.reset()
    for t in range(MAX_T):
        if render:
            env.render()
        # Execute the action
        action_probs, value = sess.run([action_probs_out, vf], {X: np.reshape(obs, [1, env.observation_space.shape[0]])})
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        value = np.asscalar(value)
        obs, reward, done, _ = env.step(action)
        experience['actions'].append(action)
        experience['values'].append(value)
        experience['obs'].append(obs)
        experience['rewards'].append(reward)
        experience['dones'].append(done)

        if done:
            return t, experience


def simulate():

    # Instantiating the learning related parameters
    render = False
    num_streaks = 0
    highest_streak = 0

    init = tf.global_variables_initializer()
    sess.run(init)
    episodes = reset_episodes()
    episodes_for_logging = {'rewards': []}
    for episode in range(NUM_EPISODES):

        def toggleRendering():
            print("Rendering now ")
            nonlocal render
            render = not render
        keyboardCommands("r", toggleRendering)
        keyboardCommands("s", lambda: tf.train.Saver().save(sess, './saves/{name}'.format(name=NAME)))

        timesteps, experience = run_episode(render)
        num_streaks = num_streaks+1 if timesteps >= SOLVED_T else 0

        advantages, returns = gae(experience['rewards'], experience['values'])

        episodes['timesteps'].append(timesteps)
        episodes_for_logging['rewards'].append(np.sum(experience['rewards']))
        episodes['actions'].append(experience['actions'])
        episodes['obs'].append(experience['obs'])
        episodes['returns'].append(returns)
        episodes['advantages'].append(advantages)
        highest_streak = max(highest_streak, num_streaks)

        if(episode > 0 and episode % 5 == 0):
            for idx in range(len(episodes['timesteps'])):
                vl, pl, _ = sess.run(
                    [value_loss, policy_loss, train],
                    {
                        X: episodes['obs'][idx],
                        RETURNS: episodes['returns'][idx],
                        ADVANTAGE: episodes['advantages'][idx],
                        ACTION: episodes['actions'][idx]
                    }
                )
            episodes = reset_episodes()
            
        if(episode > 0 and episode % 100 == 0):
            mean_reward = np.mean(episodes_for_logging['rewards'])
            max_reward = np.max(episodes_for_logging['rewards'])
            min_reward = np.min(episodes_for_logging['rewards'])
            summary = sess.run(
                merged,
                {
                    MEAN: mean_reward,
                    MAX: max_reward,
                    MIN: min_reward,
                    STREAK: highest_streak
                }
            )
            train_writer.add_summary(summary, episode)
            print("Episode %s Reward %s Max %s Min %s Streak %s" % (episode, mean_reward, max_reward, min_reward, highest_streak))
            highest_streak = 0
            episodes_for_logging = {'rewards': []}

        if EXTRA_LOG:
            pl, ai, pi, apo, ppa = sess.run(
                [policy_loss, action_indices, prob_indices, action_probs_out, prob_of_picked_action],
                {
                    X: experience['obs'],
                    ADVANTAGE: advantages,
                    ACTION: experience['actions'],
                }
            )
            print("Timesteps %s" % timesteps)
            print("Computed %s %s %s %s %s" % (pl, ai, pi, apo, ppa))
            # print("Values %s %s" % (np.round(experience['values'], 1)[0:10], np.round(experience['values'], 1)[-10:]))
            print("Rewards %s %s %s" % (np.round(experience['rewards'], 1)[0:5], np.round(experience['rewards'], 1)[-5:]))
            print("Actions %s %s %s" % (np.round(experience['actions'], 1)[0:5], np.round(experience['actions'], 1)[-5:]))
            # print("Adv %s %s" % (np.round(advantages, 1)[0:3], np.round(advantages, 1)[-3:]))


if __name__ == "__main__":
    simulate()
