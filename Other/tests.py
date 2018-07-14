import tensorflow as tf
import numpy as np

print(np.zeros(10))
# flat = tf.placeholder(tf.float32, [5])
# mean, logstd = tf.split(axis=len(flat.shape)-1, num_or_size_splits=2, value=flat)
# std = tf.exp(logstd)
# print (0.5 * tf.reduce_sum(tf.square((x - mean) / std), axis=-1) \
#                + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
#                + tf.reduce_sum(logstd, axis=-1))

ACTION = tf.placeholder(tf.int32, [None, 2])
PROBS = tf.placeholder(tf.int32, [None, 10])

action_indices = tf.range(tf.shape(ACTION)[0])
prob_indices1 = tf.concat([tf.reshape(action_indices, [-1, 1]), tf.reshape(ACTION[:, 0], [-1, 1])], axis=1)
prob_indices2 = tf.concat([tf.reshape(action_indices, [-1, 1]), tf.reshape(ACTION[:, 1], [-1, 1])], axis=1)
prob_of_picked_action1 = tf.gather_nd(PROBS, prob_indices1)
prob_of_picked_action2 = tf.gather_nd(PROBS, prob_indices2)
prob_of_picked_action = tf.add(prob_of_picked_action1, prob_of_picked_action2)

# prob_of_picked_action = tf.gather_nd(self.action_probs_out, self.ACTION)

sess = tf.Session()
alist = [[3, 4], [1, 3], [1, 2]]
probs = [
    [3, 4, 1, 2, 2, 3, 5, 4, 6, 8],
    [1, 3, 1, 2, 2, 3, 5, 4, 6, 8],
    [1, 2, 1, 2, 2, 3, 5, 4, 6, 8]
]
indices, prob_indices1, prob_indices2, prob_of_picked_action1, prob_of_picked_action2, prob_of_picked_action = sess.run(
    [action_indices, prob_indices1, prob_indices2, prob_of_picked_action1, prob_of_picked_action2, prob_of_picked_action],
    {ACTION: alist, PROBS: probs}
)
print(indices)
print(prob_indices1)
print(prob_indices2)
print(prob_of_picked_action1)
print(prob_of_picked_action2)
print(prob_of_picked_action)
