import tensorflow as tf
import numpy as np


def create_model(name, input_size, num_layers, num_conv_layers):
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
