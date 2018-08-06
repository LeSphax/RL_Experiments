from Matchmaking.matchmaking_agents.Policies.policy import Policy
import tensorflow as tf
import numpy as np

class CategoricalPd(object):
    def __init__(self, logits):
        self.logits = logits

    def neglogp(self, x):
        one_hot_actions = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
        return tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=one_hot_actions)

    def entropy(self):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)

    def sample(self):
        u = tf.random_uniform(tf.shape(self.logits))
        return tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)


class DNNPolicy(Policy):

    def __init__(self, model_function, env, reuse=False):
        self.input_shape = env.observation_space.shape
        self.output_size = env.action_space.n
        name = 'policy'

        self.X, previous_layer = model_function(name, self.input_shape, reuse)

        with tf.variable_scope(name + '/training', reuse=reuse):
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=previous_layer,
                num_outputs=self.output_size,
                activation_fn=None,
                weights_initializer=tf.constant_initializer(0.01)
            )

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

            self.params = tf.trainable_variables()
            grads = tf.gradients(self.loss, self.params)
            grads, _grad_norm = tf.clip_by_global_norm(grads, 0.5)
            self.grads_and_vars = list(zip(grads, self.params))

            optimizer = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE, epsilon=1e-5)
            self._train = optimizer.apply_gradients(self.grads_and_vars)

            from tensorflow.python import debug as tf_debug

            self.sess = tf.get_default_session()
            # self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
            init = tf.global_variables_initializer()
            self.sess.run(init)

    def get_action(self, obs):
        action, neglogp_action = self.sess.run([self.action, self.neglogp_action],
                                               {self.X: obs})

        return action, neglogp_action

    def train(self, obs, actions, neglogp_actions, advantages, clipping, learning_rate):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        entropy, loss, _ = self.sess.run(
            [self.entropy, self.loss, self._train],
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

    def get_weights(self):
        values = tf.get_default_session().run(tf.global_variables())
        return values

    def set_weights(self, weights):
        for i, value in enumerate(weights):
            value = np.asarray(value)
            tf.global_variables()[i].load(value, self.sess)