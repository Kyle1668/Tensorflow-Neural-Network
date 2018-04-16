import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def network_layer(n_features, n_dense_neurons):
    """[Example of a layer of a neural network.]
    
    Arguments:
        n_features {[int]} -- [The number of features in each vector.]
        n_dense_neurons {[int]} -- [The number of layers.]
    """

    input_data = tf.placeholder(dtype=tf.float32, shape=[None, n_features])
    weights = tf.Variable(tf.random_normal([n_features, n_dense_neurons]))
    bias = tf.Variable(tf.ones([n_dense_neurons]))
    output = tf.add(tf.matmul(input_data, weights), bias)
    activation = tf.sigmoid(output)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        layer_out = sess.run(activation, feed_dict={input_data: np.random.random([1, n_features])})
        print(layer_out)


def regression_example(starting_point, stopping_point, num_points):
    """[Example of a regression task.]
    
    Arguments:
        starting_point {[int]} -- [The lower bound of the random numbers.]
        stopping_point {[int]} -- [The upper bound of the random numbers.]
        num_points {[int]} -- [The number of points between the lower an dupper bound to be generated.]
    """

    x_data = np.linspace(starting_point, stopping_point, num_points) + np.random.uniform(-1.5, 1.5, 10)
    y_data = np.linspace(starting_point, stopping_point, num_points) + np.random.uniform(-1.5, 1.5, 10)
    plt.plot(x_data, y_data)
    plt.show()


# network_layer(5, 3)
regression_example(0, 1, 10)