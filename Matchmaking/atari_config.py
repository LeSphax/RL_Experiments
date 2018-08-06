import random

import cv2
import tensorflow as tf
import numpy as np

import gym
from Matchmaking import EnvConfiguration
from Matchmaking.wrappers.auto_reset_env import AutoResetEnv
from Matchmaking.wrappers.monitor_env import MonitorEnv
from Matchmaking.wrappers.tensorboard_vec_env import TensorboardVecEnv
from Matchmaking.wrappers.vec_env.subproc_vec_env import SubprocVecEnv
from gym import Wrapper, spaces


# Take the 210x160 RGB frames from the atari env and turn then into stacks of 4 frames with 84x84 grayscale
class ProcessStateEnv(Wrapper):
    def __init__(self, env):
        Wrapper.__init__(self, env=env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 4), dtype=np.uint8)
        self.state = None

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.state = np.append(self.state[:, :, 1:], np.expand_dims(obs, 2), axis=2)
        assert (np.shape(self.state) == (84, 84, 4))
        return self.state, rew, done, info

    def reset(self):
        obs = self.env.reset()
        self.state = np.stack([obs] * 4, axis=2)
        return self.state


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.height, self.width), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :]


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class AtariConfig(EnvConfiguration):

    def create_model(self, name, input_shape, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            X = tf.placeholder(shape=(None,) + input_shape, dtype=np.float32, name="X")

            scaled_images = tf.cast(X, tf.float32) / 255.
            previous_layer = tf.reshape(scaled_images, (-1,) + input_shape)

            activ = tf.nn.relu
            previous_layer = tf.contrib.layers.conv2d(
                inputs=previous_layer,
                num_outputs=32,
                kernel_size=8,
                padding="valid",
                activation_fn=activ,
                stride=4,
                weights_initializer=tf.orthogonal_initializer(np.sqrt(2))
            )

            previous_layer = tf.contrib.layers.conv2d(
                inputs=previous_layer,
                num_outputs=64,
                kernel_size=4,
                padding="valid",
                activation_fn=activ,
                stride=2,
                weights_initializer=tf.orthogonal_initializer(np.sqrt(2))
            )

            previous_layer = tf.contrib.layers.conv2d(
                inputs=previous_layer,
                num_outputs=64,
                kernel_size=3,
                padding="valid",
                activation_fn=activ,
                stride=1,
                weights_initializer=tf.orthogonal_initializer(np.sqrt(2))
            )

            total_size = np.prod([v.value for v in previous_layer.get_shape()[1:]])
            previous_layer = tf.reshape(previous_layer, [-1, total_size])

            for idx in range(1):
                hidden_layer = tf.contrib.layers.fully_connected(
                    inputs=previous_layer,
                    num_outputs=512,
                    activation_fn=activ,
                    weights_initializer=tf.orthogonal_initializer(np.sqrt(2))
                )
                previous_layer = hidden_layer
            return X, previous_layer

    def _parameters(self):
        return {
            "seed": 1,
            "decay": False,
            "num_env": 8,
            "batch_size": 128,
            "nb_epochs": 4,
            "nb_minibatch": 4,
            "clipping": 0.1,
            "learning_rate": 0.00025,
            "total_timesteps": int(80e6),
        }

    @property
    def env_name(self):
        return "Breakout-v0"

    def make_env(self, proc_idx=0, save_path=None):
        env = gym.make(self.env_name)

        env.seed(self.parameters.seed + proc_idx)
        env = MonitorEnv(env)
        env = EpisodicLifeEnv(env)
        env = WarpFrame(env)
        env = ProcessStateEnv(env)

        return env

    def make_vec_env(self, save_path=None, renderer=False):
        def make_env_fn(proc_idx):
            def _thunk():
                return self.make_env(proc_idx, save_path)

            return _thunk

        if renderer:
            return make_env_fn(1)
        else:
            venv = SubprocVecEnv([make_env_fn(i) for i in range(self.parameters.num_env)])
            return TensorboardVecEnv(venv, save_path)
