import numpy as np

from collections import OrderedDict

from pyGPGO.GPGO import GPGO
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.acquisition import Acquisition
from pyGPGO.covfunc import squaredExponential

from bayes_opt import BayesianOptimization
from funcgenerator import get_function
import matplotlib.pyplot as plt


def plot_f(x_values, y_values, f):
    # Plot example borrowed from {pyGPGO github examples}
    # https://github.com/hawk31/pyGPGO/blob/master/examples/example2d.py

    z = np.zeros((len(x_values), len(y_values)))
    for i in range(len(x_values)):
        for j in range(len(y_values)):
            z[i, j] = f(x_values[i], y_values[j])
    plt.imshow(z.T, origin='lower', extent=[np.min(x_values), np.max(x_values), np.min(y_values), np.max(y_values)])
    plt.colorbar()
    plt.show()


def plot2dgpgo(gpgo):

    # Plot example borrowed from {pyGPGO github examples}
    # https://github.com/hawk31/pyGPGO/blob/master/examples/example2d.py

    tested_X = gpgo.GP.X
    n = 100
    r_x, r_y = gpgo.parameter_range[0], gpgo.parameter_range[1]
    x_test = np.linspace(r_x[0], r_x[1], n)
    y_test = np.linspace(r_y[0], r_y[1], n)
    z_hat = np.empty((len(x_test), len(y_test)))
    z_var = np.empty((len(x_test), len(y_test)))
    ac = np.empty((len(x_test), len(y_test)))
    for i in range(len(x_test)):
        for j in range(len(y_test)):
            res = gpgo.GP.predict([x_test[i], y_test[j]])
            z_hat[i, j] = res[0]
            z_var[i, j] = res[1][0]
            ac[i, j] = -gpgo._acqWrapper(np.atleast_1d([x_test[i], y_test[j]]))
    fig = plt.figure()
    a = fig.add_subplot(2, 2, 1)
    a.set_title('Posterior mean')
    plt.imshow(z_hat.T, origin='lower', extent=[r_x[0], r_x[1], r_y[0], r_y[1]])
    plt.colorbar()
    plt.plot(tested_X[:, 0], tested_X[:, 1], 'wx', markersize=10)
    a = fig.add_subplot(2, 2, 2)
    a.set_title('Posterior variance')
    plt.imshow(z_var.T, origin='lower', extent=[r_x[0], r_x[1], r_y[0], r_y[1]])
    plt.plot(tested_X[:, 0], tested_X[:, 1], 'wx', markersize=10)
    plt.colorbar()
    a = fig.add_subplot(2, 2, 3)
    a.set_title('Acquisition function')
    plt.imshow(ac.T, origin='lower', extent=[r_x[0], r_x[1], r_y[0], r_y[1]])
    plt.colorbar()
    gpgo._optimizeAcq(method='L-BFGS-B', n_start=500)
    plt.plot(gpgo.best[0], gpgo.best[1], 'gx', markersize=15)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    x1 = np.concatenate(
        (np.random.rand(50, 1) - 3, np.random.rand(50, 1) - 2, np.random.rand(50, 1), -1 * np.random.rand(50, 1),
         np.random.rand(50, 1) + 1, np.random.rand(50, 1) + 2))
    x2 = np.concatenate(
        (np.random.rand(50, 1) - 3, np.random.rand(50, 1) - 2, np.random.rand(50, 1), -1 * np.random.rand(50, 1),
         np.random.rand(50, 1) + 1, np.random.rand(50, 1) + 2))

    plot_f(x1, x2, get_function)

    np.random.seed(20)
    sexp = squaredExponential()
    gp = GaussianProcess(sexp)
    acq = Acquisition(mode='ExpectedImprovement')

    param = OrderedDict()
    param['x'] = ('cont', [-3, 3])
    param['y'] = ('cont', [-3, 3])

    gpgo = GPGO(gp, acq, get_function, param, n_jobs=-1)
    gpgo._firstRun()

    for item in range(13):
        # if item >= 11:
        plot2dgpgo(gpgo)
        gpgo.updateGP()
    print(gpgo.getResult())
