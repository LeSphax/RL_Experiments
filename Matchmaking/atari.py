import tensorflow as tf
import numpy as np
from gym import Wrapper, spaces


# Take the 210x160 RGB frames from the atari env and turn then into stacks of 4 frames with 84x84 grayscale
class ProcessStateEnv(Wrapper):
    def __init__(self, env):
        Wrapper.__init__(self, env=env)
        self.sp = StateProcessor()
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 4), dtype=np.uint8)
        self.state = None

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        obs = self.sp.process(obs)
        np.append(self.state[:, :, 1:], np.expand_dims(obs, 2), axis=2)
        return self.state, rew, done, info

    def reset(self):
        obs = self.sp.process(self.env.reset())
        self.state = np.stack([obs] * 4, axis=2)
        return self.state


class StateProcessor():
    """
    Processes a raw Atari image. Resizes it and converts it to grayscale.
    """

    def __init__(self):
        # Build the Tensorflow graph
        with tf.variable_scope("state_processor", reuse=True):
            self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(
                self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def process(self, state):
        """
        Args:
            sess: A Tensorflow session object
            state: A [210, 160, 3] Atari RGB State

        Returns:
            A processed [84, 84, 1] state representing grayscale values.
        """
        sess = tf.get_default_session()
        return sess.run(self.output, {self.input_state: state})


def create_model(name, input_shape, num_layers, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        X = tf.placeholder(shape=(None,) + input_shape, dtype=np.float32, name="X")

        previous_layer = tf.cast(X, tf.float32) / 255.
        previous_layer = tf.reshape(previous_layer, (-1,) + input_shape)

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
        for idx in range(num_layers):
            hidden_layer = tf.contrib.layers.fully_connected(
                inputs=previous_layer,
                num_outputs=256,
                activation_fn=activ,
                weights_initializer=tf.orthogonal_initializer(np.sqrt(2))
            )
            previous_layer = hidden_layer
        return X, previous_layer
