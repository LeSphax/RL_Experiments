import random
from collections import deque

import cv2
import tensorflow as tf
import numpy as np

import gym
from Matchmaking import EnvConfiguration
from Matchmaking.wrappers.auto_reset_env import AutoResetEnv
from Matchmaking.wrappers.monitor_env import MonitorEnv
from Matchmaking.wrappers.tensorboard_vec_env import TensorboardVecEnv
from Matchmaking.wrappers.vec_env.dummy_vec_env import DummyVecEnv
from Matchmaking.wrappers.vec_env.subproc_vec_env import SubprocVecEnv
from Matchmaking.wrappers.vec_env.vec_frame_stack import VecFrameStack
from gym import Wrapper, spaces

# Take frames from atari and return stack of frames
from gym.wrappers import Monitor


class ProcessStateEnv(Wrapper):
    def __init__(self, env, stack_size=4):
        Wrapper.__init__(self, env=env)
        self.stack_size = stack_size
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, self.stack_size), dtype=np.uint8)
        self.state = None

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.state = np.append(self.state[:, :, 1:], np.expand_dims(obs, 2), axis=2)
        assert (np.shape(self.state) == (84, 84, self.stack_size))
        return self.state, rew, done, info

    def reset(self):
        obs = self.env.reset()
        self.state = np.stack([obs] * self.stack_size, axis=2)
        return self.state


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


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
        self.was_real_done = True

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


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class BreakoutNoFrameskipConfig(EnvConfiguration):

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
        return "BreakoutNoFrameskip-v4"

    def make_env(self, proc_idx=0, save_path=None, renderer=False):
        env = gym.make(self.env_name)

        env.seed(self.parameters.seed + proc_idx)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = MonitorEnv(env)
        env = EpisodicLifeEnv(env)
        env = FireResetEnv(env)
        env = WarpFrame(env)
        env = ClipRewardEnv(env)
        if save_path:
            env = Monitor(env, directory=save_path and str(proc_idx), video_callable=lambda x: True, resume=True)

        return env

    def make_vec_env(self, save_path=None, renderer=False):
        if renderer:
            venv = DummyVecEnv([self.make_env_fn()])
        else:
            venv = SubprocVecEnv([self.make_env_fn(i, save_path) for i in range(self.parameters.num_env)])
            venv = TensorboardVecEnv(venv, save_path)

        venv = VecFrameStack(venv, 4)
        print(venv.observation_space.shape)
        return venv
