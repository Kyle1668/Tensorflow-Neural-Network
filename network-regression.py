import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


def get_data(lower_bound, upper_bound, amount):
    """[Returns a pandas dataframe based on user configuration.]
    
    Arguments:
        lower_bound {[int]} -- [The lower bound of the random generated values.]
        upper_bound {[int]} -- [The upper bound of the random generated values.]
        amount {[int]} -- [The amount of random generated values.]
    
    Returns:
        [pandas data frame] -- [The data frame concisting of the X and Y columns]
    """

    x_data = np.linspace(lower_bound, upper_bound, amount)
    noise = np.random.randn(len(x_data))
    y_true = (0.5 * x_data) + 5 + noise

    x_data_frame = pd.DataFrame(data=x_data, columns=["X Data"])
    y_data_frame = pd.DataFrame(data=y_true, columns=["Y Data"])
    
    return pd.concat([x_data_frame, y_data_frame], axis=1)


def set_optimizer(target_sum, rate):
    error = tf.reduce_sum(target_sum)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=rate)
    return optimizer.minimize(error)


def run_network(batch_size):
    x_data = np.linspace(0.0, 10, 1000000)
    noise = np.random.randn(len(x_data))
    y_true = (0.5 * x_data) + 5 + noise
    
    m_slope = tf.Variable(0.81)
    b_intercept = tf.Variable(0.17)

    xph = tf.placeholder(tf.float32, [batch_size])
    yph = tf.placeholder(tf.float32, [batch_size])

    y_model = m_slope*xph + b_intercept
    train = set_optimizer(tf.square(yph - y_model), 0.001)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        batches = 1000

        for i in range(batches):
            rand_index = np.random.randint(len(x_data), size=batch_size)

            feed = {
                xph: x_data[rand_index], 
                yph: y_true[rand_index]
            }

            sess.run(train, feed_dict=feed)
        
        model_m, model_b = sess.run([m_slope, b_intercept])

        print("Result: " + str(model_m))

        y_hat = model_m**x_data + model_b

        plt.plot(x_data, y_hat)
        plt.show()


run_network(8)