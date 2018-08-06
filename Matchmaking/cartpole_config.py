import random

import numpy as np
import tensorflow as tf

import gym
from Matchmaking import EnvConfiguration
from Matchmaking.atari_config import WarpFrame, ProcessStateEnv
from Matchmaking.wrappers.auto_reset_env import AutoResetEnv
from Matchmaking.wrappers.monitor_env import MonitorEnv
from Matchmaking.wrappers.normalize_env import NormalizeEnv
from Matchmaking.wrappers.tensorboard_vec_env import TensorboardVecEnv
from Matchmaking.wrappers.vec_env.dummy_vec_env import DummyVecEnv
from Matchmaking.wrappers.vec_env.subproc_vec_env import SubprocVecEnv
from Matchmaking.wrappers.vec_env.vec_normalize import VecNormalize


class CartPoleConfig(EnvConfiguration):

    def create_model(self, name, input_shape, reuse=False):
        input_size = input_shape[0]
        with tf.variable_scope(name, reuse=reuse):
            X = tf.placeholder(shape=[None, input_size], dtype=np.float32, name="X")

            previous_layer = X

            for idx in range(2):
                hidden_layer = tf.contrib.layers.fully_connected(
                    inputs=previous_layer,
                    num_outputs=64,
                    activation_fn=tf.nn.tanh,
                    weights_initializer=tf.orthogonal_initializer(np.sqrt(2))
                )
                previous_layer = hidden_layer
            return X, previous_layer

    def _parameters(self):
        return {
            "seed": 1,
            "decay": True,
            "num_env": 8,
            "batch_size": 64,
            "nb_epochs": 4,
            "nb_minibatch": 4,
            "clipping": 0.1,
            "learning_rate": 2.5e-4,
            "total_timesteps": 1000000,
        }

    @property
    def env_name(self):
        return "CartPole-v1"

    def make_env(self, renderer=False):
        env = gym.make(self.env_name)

        env.seed(self.parameters.seed)

        env = MonitorEnv(env)

        return env

    def make_vec_env(self, save_path=None, renderer=False):
        if renderer:
            venv = DummyVecEnv([self.make_env_fn()])
            venv = VecNormalize(venv, reuse=True)

        else:
            venv = SubprocVecEnv([self.make_env_fn(i) for i in range(self.parameters.num_env)])
            venv = TensorboardVecEnv(venv, save_path)
            venv = VecNormalize(venv)

        return venv
