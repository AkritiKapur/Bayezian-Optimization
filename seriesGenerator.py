import numpy as np
import matplotlib.pyplot as plt

# probability of changing mode.
beta = 0.2
alpha1 = -0.6
alpha2 = 0.4
sigma = 0.2


def generate_data(T):
    """
        Generates time series data for two elements
        dependent on the following equations:
        y2(t) = y1(t-1) + μ1
        y1(t) = α1 y1(t-1) + α2 y2(t-1) + μ2

        and observation:
        z(t) = y1(t) + μ3

        where  μi ~ Normal(0,σ2)
    :param T time interval till the data has to be generated
    :return: {Dict of lists} Data generated by the model
             at each time step
    """

    # initialize random values for the two variables at timestep 0
    y1 = {0: np.random.random_sample()}
    y2 = {0: np.random.random_sample()}

    muu1, muu2, muu3 = np.random.normal(loc=0, scale=sigma, size=3)
    z = {0: y1[0] + muu3}

    # populate data according to the equations
    for t in range(1, T):
        y2[t] = y1[t-1] + muu1
        y1[t] = alpha1 * y1[t-1] + alpha2 * y2[t-1] + muu2

        z[t] = y1[t] + muu3
        # muu1, muu2, muu3 = np.random.normal(loc=0, scale=sigma, size=3)

    return {"y1": y1, "y2": y2, "z": z}


def is_change_mode():
    """
    :return: True if the mode is to be changed else False
    """
    return np.random.random() <= beta


def get_mode(n_modes):
    """
        Choose a mode from a uniform distribution
    :param n_modes: number of modes to choose from
    :return: Mode ID
    """
    return np.random.choice(n_modes)


def generate_data_switched_model(T):
    """
        Generates time series data for two elements
        dependent on the following equations:
        y2(t) = y1(t-1) + μ1
        y1(t) = α1 y1(t-1) + α2 y2(t-1) + μ2

        and observation:
        z(t) = y1(t) + μ3

        where  μi ~ Normal(0,σ2)

        Model is a switched linear dynamical system with 3 modes.
        Each mode has an associated set of {α1,α2} parameters.
        At each time with some small probability β, the mode m,
        is re-drawn from the set {1,2,3} with uniform probability.

    :param T: T time interval till the data has to be generated
    :return: {Dict of lists} Data generated by the model
             at each time step.
    """
    # create alpha dict with key as mode ID
    # and alpha1 and alpha2 as the values
    # TODO: Tune these parameters
    alphas = {0: [0.3, -0.5], 1: [0.1, 0.4], 2: [0.8, 0.2]}
    transition_matrix = np.zeros((3, 3, T))
    count_modes = np.zeros(3)

    # Initialize Sigma
    sigma = 0.7

    # Pick the first mode
    mode = get_mode(3)
    modes = [mode]
    muu1, muu2, muu3 = np.random.normal(loc=0, scale=sigma, size=3)

    # initialize random values for the two variables at timestep 0
    y1 = {0: np.random.random_sample()}
    y2 = {0: np.random.random_sample()}

    z = {0: y1[0] + muu3}

    for t in range(1, T):
        y2[t] = y1[t - 1] + muu1
        y1[t] = alphas[mode][0] * y1[t - 1] + alphas[mode][1] * y2[t - 1] + muu2

        z[t] = y1[t] + muu3

        if is_change_mode():
            mode = get_mode(3)

        modes.append(mode)

    return {"y1": y1, "y2": y2, "z": z, "mode": modes}


def plot_time_series_data(data, mode=False):
    """
        Plots data generated by the above process
    :param data: Data to be plotted
    :return: None
    """
    x = np.arange(len(data['y1']))

    plt.plot(x, data['y1'].values(), color="green")
    # plt.show()
    plt.plot(x, data['y2'].values(), color="purple")
    # plt.show()
    plt.plot(x, data['z'].values(), color="orange")
    if mode:
        plt.plot(x, data['mode'], color="pink")
    plt.show()


if __name__ == '__main__':
    # Generate data for time stamp
    data = generate_data(200)

    # Plot the data
    plot_time_series_data(data)

    # plot data for switched dynamic model
    data = generate_data_switched_model(200)

    # Plot the data
    plot_time_series_data(data, mode=True)
