import tensorflow as tf
import numpy as np


def create_model(name, input_shape, num_layers, num_conv_layers, reuse=False):
    input_size = input_shape[0]
    with tf.variable_scope(name, reuse=False):
        X = tf.placeholder(shape=[None, input_size], dtype=np.float32, name="X")

        previous_layer = X
        for idx in range(num_conv_layers):
            previous_layer = tf.reshape(X, [-1, input_size, 1])
            previous_layer = tf.layers.conv1d(
                inputs=previous_layer,
                filters=16,
                kernel_size=[3],
                padding="same",
                activation=tf.nn.tanh)
            previous_layer = tf.reshape(previous_layer, [-1, 16 * input_size])
        for idx in range(num_layers):
            hidden_layer = tf.contrib.layers.fully_connected(
                inputs=previous_layer,
                num_outputs=64,
                activation_fn=tf.nn.tanh,
                weights_initializer=tf.orthogonal_initializer(np.sqrt(2))
            )
            previous_layer = hidden_layer
        return X, previous_layer


def create_atari_model(name, input_shape, num_layers, num_conv_layers, reuse=False):
    nb_filters = 1
    with tf.variable_scope(name, reuse=reuse):
        X = tf.placeholder(shape=(None,) + input_shape, dtype=np.float32, name="X")

        previous_layer = tf.reshape(X, (-1,) + input_shape)
        previous_layer = tf.contrib.layers.max_pool2d(
            inputs=previous_layer,
            kernel_size=2,
            stride=2,
            padding='valid'
        )
        previous_layer = tf.contrib.layers.max_pool2d(
            inputs=previous_layer,
            kernel_size=2,
            stride=2,
            padding='valid'
        )
        for idx in range(num_conv_layers):
            previous_layer = tf.contrib.layers.conv2d(
                inputs=previous_layer,
                num_outputs=nb_filters,
                kernel_size=[3, 3],
                padding="same",
                activation_fn=tf.nn.relu
            )

        total_size = np.prod([v.value for v in previous_layer.get_shape()[1:]])
        previous_layer = tf.reshape(previous_layer, [-1, total_size])
        for idx in range(num_layers):
            hidden_layer = tf.contrib.layers.fully_connected(
                inputs=previous_layer,
                num_outputs=64,
                activation_fn=tf.nn.tanh,
                weights_initializer=tf.orthogonal_initializer(np.sqrt(2))
            )
            previous_layer = hidden_layer
        return X, previous_layer
