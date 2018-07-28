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


