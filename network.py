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


def nn_example(starting_point, stopping_point, num_points):
    """[Function to solve a linear quadratin equation.]
    
    Arguments:
        starting_point {[int]} -- [The lower bound of the random numbers.]
        stopping_point {[int]} -- [The upper bound of the random numbers.]
        num_points {[int]} -- [The number of points between the lower an dupper bound to be generated.]
    """

    x_data = np.linspace(starting_point, stopping_point, num_points) + np.random.uniform(-1.5, 1.5, 10)
    y_data = np.linspace(starting_point, stopping_point, num_points) + np.random.uniform(-1.5, 1.5, 10)

    rands = np.random.rand(2)
    m = tf.Variable(rands[0])
    b = tf.Variable(rands[1])
    error = 0

    for x, y in zip(x_data, y_data):
        y_hat = m*x + b    
        error += (y - y_hat)**2
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train = optimizer.minimize(error)
        
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        training_steps = 50000

        for i in range(training_steps):
            print(sess.run(error))
            sess.run(train)
            
        final_slope, final_inetcept = sess.run([m, b])
        x_test = np.linspace(-1, 11, 10)
        y_pred_plot = final_slope * x_test + final_inetcept


        plt.plot(x_test, y_pred_plot)
        plt.plot(x_data, y_data)
        plt.show()



# network_layer(5, 3)
nn_example(0, 10, 10)