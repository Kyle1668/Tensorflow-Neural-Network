import numpy as np
import tensorflow as tf

n_features = 5
n_dense_neurons = 3

input_data = tf.placeholder(dtype=tf.float32, shape=[None, n_features])
weights = tf.Variable(tf.random_normal([n_features, n_dense_neurons]))
bias = tf.Variable(tf.ones([n_dense_neurons]))
output = tf.add(tf.matmul(input_data, weights), bias)
activation = tf.sigmoid(output)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    layer_out = sess.run(activation, feed_dict={input_data: np.random.random([1, n_features])})
    print(layer_out)
