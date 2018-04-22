from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture


def generate_non_linear_function_output(x1, x2):
    """
        Generates f(x)/ output for the input vector
        according to the non linear function specified.
    :return: f(x), output vector
    """

    f = 0.5 * x1 + 0.3 * x2  # Linear combination of inputs
    # Fit a Gaussian mixture with EM using 4 components

    obs = np.concatenate((20 * np.random.randn(300, 1) + 5, 50 * np.random.randn(200, 1) + 10,
                          100 * np.random.randn(300, 1) + 9, 170 * np.random.randn(200, 1) + 50))
    # gmm = mixture.GaussianMixture(n_components=5, covariance_type='full').fit(f)

    g = mixture.GaussianMixture(n_components=4)
    g.fit(obs)
    pred = g.predict(f.reshape(len(f), 1))
    print(Counter(pred.tolist()))

    return pred


def plot_function(input, output):
    """
        Generates input-output plot for the 2D function
    :param input:
    :param output:
    :return:
    """
    pass

if __name__ == '__main__':
    x1 = np.arange(200)
    x2 = np.arange(200)
    x1 = np.random.choice(x1, 50)
    x2 = np.random.choice(x2, 50)
    output_vector = generate_non_linear_function_output(x1, x2)
    # plot_function(input_vector, output_vector)
