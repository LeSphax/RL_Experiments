from gym.spaces import Discrete, Box
import math
import matchmaking_agents.agents
from matchmaking_agents.Values.value import MatchmakingValue
import tensorflow as tf
import numpy as np


def observation_input(ob_space, batch_size=None, name='Ob'):
    '''
    Build observation input with encoding depending on the 
    observation space type
    Params:
    
    ob_space: observation space (should be one of gym.spaces)
    batch_size: batch size for input (default is None, so that resulting input placeholder can take tensors with any batch size)
    name: tensorflow variable name for input placeholder

    returns: tuple (input_placeholder, processed_input_tensor)
    '''
    if isinstance(ob_space, Discrete):
        input_x  = tf.placeholder(shape=(batch_size,), dtype=tf.int32, name=name)
        processed_x = tf.to_float(tf.one_hot(input_x, ob_space.n))
        return input_x, processed_x

    elif isinstance(ob_space, Box):
        input_shape = (batch_size,) + ob_space.shape
        input_x = tf.placeholder(shape=input_shape, dtype=ob_space.dtype, name=name)
        processed_x = tf.to_float(input_x)
        return input_x, processed_x

    else:
        raise NotImplementedError


def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init

def fc(x, scope, nh, *, init_scale=1.0, init_bias=0.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias))
        return tf.matmul(x, w)+b


class DNNValue(MatchmakingValue):

    def __init__(self, env, num_layers=1):
        self.input_size = env.observation_space.shape[0]
        self.create_model(env, num_layers)

    def create_model(self, env, num_layers):
        with tf.variable_scope("value", reuse=False):
            self.X, processed_x = observation_input(env.observation_space, None)
            activ = tf.tanh
            processed_x = tf.layers.flatten(processed_x)

            previous_layer = processed_x
            for idx in range(num_layers):
                hidden_layer = activ(fc(previous_layer, 'value_fc'+str(idx), nh=64, init_scale=np.sqrt(2)))
                previous_layer = hidden_layer

            self.value = fc(previous_layer, 'vf', 1)[:,0]


        self.OLD_VALUES = tf.placeholder(tf.float32, [None])
        self.RETURNS = tf.placeholder(tf.float32, [None])
        self.LEARNING_RATE = tf.placeholder(tf.float32, ())
        self.CLIPPING = tf.placeholder(tf.float32, ())

        value_clipped = self.OLD_VALUES + tf.clip_by_value(self.value - self.OLD_VALUES, -self.CLIPPING,  self.CLIPPING)
        losses1 = tf.square(self.value - self.RETURNS)
        losses2 = tf.square(value_clipped - self.RETURNS)
        self.loss = .5 * tf.reduce_mean(tf.maximum(losses1, losses2))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE)
        self.train = optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def get_value(self, obs):
        # print(obs)
        return self.sess.run(self.value, {self.X: np.reshape(obs, [1, self.input_size])})

    def train_model(self, obs, values, returns, clipping, learning_rate):
        values, loss, _ = self.sess.run(
            [self.value, self.loss, self.train],
            {
                self.X: obs,
                self.OLD_VALUES: values,
                self.RETURNS: returns,
                self.CLIPPING: clipping,
                self.LEARNING_RATE: learning_rate,
            }
        )

        return loss
