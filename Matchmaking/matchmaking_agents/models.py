import tensorflow as tf
import numpy as np


def create_model(name, input_shape, num_layers, reuse=False):
    input_size = input_shape[0]
    with tf.variable_scope(name, reuse=False):
        X = tf.placeholder(shape=[None, input_size], dtype=np.float32, name="X")

        previous_layer = X

        for idx in range(num_layers):
            hidden_layer = tf.contrib.layers.fully_connected(
                inputs=previous_layer,
                num_outputs=64,
                activation_fn=tf.nn.tanh,
                weights_initializer=tf.orthogonal_initializer(np.sqrt(2))
            )
            previous_layer = hidden_layer
        return X, previous_layer


def create_atari_model(name, input_shape, num_layers, reuse=False):
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
                num_outputs=16,
                activation_fn=activ,
                weights_initializer=tf.orthogonal_initializer(np.sqrt(2))
            )
            previous_layer = hidden_layer
        return X, previous_layer
