import numpy as np
import tensorflow as tf

n_features = 5
n_dense_neurons = 3

x = tf.placeholder(dtype=tf.float32, shape=(None, n_features))
w = tf.Variable(tf.random_normal([n_features, n_dense_neurons]))
bias_term = tf.Variable(tf.ones([n_dense_neurons]))

# Post Operations
xw = tf.matmul(x, w)
z = tf.add(xw, bias_term)
activation = tf.nn.sigmoid(z)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    layer_out = sess.run(activation, feed_dict={x: np.random.random([1, n_features])})
    print(layer_out)
