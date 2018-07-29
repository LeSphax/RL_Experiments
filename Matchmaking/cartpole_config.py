from Matchmaking import EnvConfiguration
import tensorflow as tf
import gym
import random
import numpy as np

from Matchmaking.wrappers.auto_reset_env import AutoResetEnv
from Matchmaking.wrappers.normalize_env import NormalizeEnv
from Matchmaking.wrappers.tensorboard_env import TensorboardEnv


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
            "batch_size": 128,
            "nb_epochs": 4,
            "nb_minibatch": 4,
            "clipping": 0.1,
            "learning_rate": 0.00025,
            "total_timesteps": 100000,
        }

    @property
    def env_name(self):
        return "CartPole-v1"

    def make_env(self, save_path=None, reuse_wrappers=False):
        env = gym.make(self.env_name)

        env.seed(self.parameters.seed)
        random.seed(self.parameters.seed)

        if save_path is not None:
            env = TensorboardEnv(env, save_path)
        env = AutoResetEnv(env, 500)
        env = NormalizeEnv(env, reuse=reuse_wrappers)

        return env
