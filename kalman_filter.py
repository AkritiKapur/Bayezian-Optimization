import numpy as np
from seriesGenerator import generate_data, alpha2, alpha1


def predict():
    pass


def update():
    pass


def kalman_filter(zs):

    # initialize Kalman values to something Appropriate
    x = np.array([0.1, 0.2])
    F = np.array([[alpha1, 1], [alpha2, 0]])
    H = np.array([1, 0])
    R = np.array([0.1])
    P = np.array([[0.5, 1], [0.5, 0]])
    Q = np.array([0.4, 0.2])

    for z in zs:
        pass
