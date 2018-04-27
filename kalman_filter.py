import matplotlib.pyplot as plt
import numpy as np
from seriesGenerator import generate_data, alpha2, alpha1


def predict(x, P, F, Q):
    x = np.dot(x, F)
    P = np.dot(np.dot(F, P), F.T) + Q

    return x, P


def update(z, x, H, P, R):
    y = z - np.dot(H, x)
    K = np.dot(P, H) / (np.dot(np.dot(H, P), H.T) + R)
    x = x + np.dot(K, y)
    P = np.dot(([[1, 0], [0, 1]] - np.dot(K, H)), P)

    return x, P


def kalman_filter(zs):

    # initialize Kalman values to something Appropriate
    x = np.array([0.1, 0.2])

    # state Transform matrix
    F = np.array([[alpha1, 1], [alpha2, 0]])

    # Conversion from state to observation
    H = np.array([1, 0])

    # noise in observation
    R = np.array([0.1])

    # Variance of state
    P = np.array([[0.5, 1], [0.5, 0]])

    # noise in state
    Q = np.array([[0.4, 0], [0, 0.2]])

    xs = []
    cov = []
    zps = []
    for z in zs:
        x, P = predict(x, P, F, Q)
        x, P = update(z, x, H, P, R)
        zp = np.dot(H, x)
        zps.append(zp)
        xs.append(x)
        cov.append(cov)

    return xs, cov, zps


def plot_prediction():
    true_data = generate_data(100)
    xs, covs, zps = kalman_filter(true_data['z'].values())

    xs = np.array(xs)
    x = np.arange(100)
    x1 = xs[:, 0]
    x2 = xs[:, 1]
    print(x1.shape)
    plt.plot(x, zps, color="green")
    plt.plot(x, true_data['z'].values(), color="red")
    plt.show()


if __name__ == '__main__':
    plot_prediction()
