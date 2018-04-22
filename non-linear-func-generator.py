from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def generate_non_linear_function_output(x1, x2):
    """
        Generates f(x)/ output for the input vector
        according to the non linear function specified.
    :return: f(x), output vector
    """
    x1, x2 = np.meshgrid(x1, x2)
    f = np.cos(x1) * np.cos(x2) * np.exp(- (x1 * x1) - (x2 + 1) * (x2 + 1))
    f2 = 10 * (x1 / 5 - pow(x1, 3) - pow(x2, 5)) * np.exp(- (x1 * x1) - (x2 * x2))
    f3 = 1 / 3 * np.exp(- (x1 + 1) * (x1 + 1) - (x2 * x2))
    f = f + f2 + f3
    return f


def plot_function(x1, x2, out):
    """
        Generates input-output plot for the 2D function
    :param input:
    :param output:
    :return:
    """
    # Plot the surface.
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x1, x2 = np.meshgrid(x1, x2)
    surf = ax.plot_surface(x1, x2, out, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    plt.show()

if __name__ == '__main__':
    x1 = np.concatenate((np.random.rand(50, 1) - 3, np.random.rand(50, 1) - 2, np.random.rand(50, 1), -1 * np.random.rand(50, 1),
                         np.random.rand(50, 1) + 1, np.random.rand(50, 1) + 2))
    x2 = np.concatenate((np.random.rand(50, 1) - 3, np.random.rand(50, 1) - 2, np.random.rand(50, 1), -1 * np.random.rand(50, 1),
                         np.random.rand(50, 1) + 1, np.random.rand(50, 1) + 2))
    output_vector = generate_non_linear_function_output(x1, x2)
    plot_function(x1, x2, output_vector)
