import tensorflow as tf
import numpy as np

ACTION = tf.placeholder(tf.int32, [None])
Probs = tf.placeholder(tf.float32, [None, 2])

# print(sess.run(tf.concat([[0,0,0],[1,1,1]], axis=1)))
indices = tf.range(tf.shape(ACTION)[0])
concat = tf.concat([tf.reshape(indices, [-1,1]),tf.reshape(ACTION, [-1,1])], axis=1)

prob_of_picked_action = tf.gather_nd(Probs,  concat)
sess = tf.Session()

print(sess.run([concat, prob_of_picked_action], {ACTION: [0,0,1,1,1,1], Probs: [[0.3,0.2],[0.2,0.1],[0.2,0.1],[0.2,0.1],[0.2,0.1], [0.4,0.6]]}))