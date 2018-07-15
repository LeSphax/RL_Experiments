from gym.spaces import Discrete, Box
import math
from matchmaking_agents.Policies.policy import Policy
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

class CategoricalPd(object):
    def __init__(self, logits):
        self.logits = logits
    def neglogp(self, x):
        one_hot_actions = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
        return tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=one_hot_actions)
    def entropy(self):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keep_dims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keep_dims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)
    def sample(self):
        u = tf.random_uniform(tf.shape(self.logits))
        return tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)

class DNNPolicy(Policy):

    def __init__(self, env, num_layers=1):
        self.input_size = env.observation_space.shape[0]
        self.output_size = env.action_space.n
        self.create_model(env, num_layers)

    def create_model(self, env, num_layers):
        with tf.variable_scope("policy", reuse=False):
            self.X, processed_x = observation_input(env.observation_space, None)
            activ = tf.tanh
            processed_x = tf.layers.flatten(processed_x)

            previous_layer = processed_x
            for idx in range(num_layers):
                hidden_layer = activ(fc(previous_layer, 'pi_fc'+str(idx), nh=64, init_scale=np.sqrt(2)))
                previous_layer = hidden_layer

            self.output_layer = fc(previous_layer, 'pi', self.output_size, init_scale=0.01, init_bias=0.0)

            self.probability_distribution = CategoricalPd(self.output_layer)

            self.action = self.probability_distribution.sample()
            self.neglogp_action = self.probability_distribution.neglogp(self.action)

        self.ADVANTAGES = tf.placeholder(tf.float32, [None])
        self.ACTIONS = tf.placeholder(tf.int32, [None])
        self.OLDNEGLOGP_ACTIONS = tf.placeholder(tf.float32, [None])
        self.LEARNING_RATE = tf.placeholder(tf.float32, ())
        self.CLIPPING = tf.placeholder(tf.float32, ())

        self.new_neglogp_action = self.probability_distribution.neglogp(self.ACTIONS)

        self.entropy = tf.reduce_mean(self.probability_distribution.entropy())

        ratio = tf.exp(self.OLDNEGLOGP_ACTIONS - self.new_neglogp_action)
        pg_losses = -self.ADVANTAGES * ratio
        pg_losses2 = -self.ADVANTAGES * tf.clip_by_value(ratio, 1.0 - self.CLIPPING, 1.0 + self.CLIPPING)
        self.loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2)) - self.entropy * 0.01
        optimizer = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE)
        self.train = optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def get_action(self, obs):
        action, neglogp_action = self.sess.run([self.action, self.neglogp_action], {self.X: np.reshape(obs, [1, self.input_size])})

        return action[0], neglogp_action[0]

    def train_model(self, obs, actions, neglogp_actions, advantages, clipping, learning_rate):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        entropy, loss, _ = self.sess.run(
            [self.entropy, self.loss, self.train],
            {
                self.X: obs,
                self.ACTIONS: actions,
                self.OLDNEGLOGP_ACTIONS: neglogp_actions,
                self.ADVANTAGES: advantages,
                self.CLIPPING: clipping,
                self.LEARNING_RATE: learning_rate,
            }
        )
        return entropy, loss
