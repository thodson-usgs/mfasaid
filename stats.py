import numpy as np


def calc_plotting_position(x, a=0.5):
    """

    :param x:
    :param a:
    :return:
    """

    x = np.asarray(x)

    Nx = x.shape[0]

    sorted_index = np.argsort(x)

    rank = np.zeros(Nx, int)
    rank[sorted_index] = np.arange(Nx) + 1

    pp = (rank - a) / (Nx + 1 - 2 * a)

    return pp


def calc_quantile(x, q):
    """

    :param x:
    :param q:
    :return:
    """

    pp = calc_plotting_position(x)

    sorted_index = np.argsort(x)

    xp = x[sorted_index]
    pp = pp[sorted_index]

    quantile = np.interp(q, pp, xp)

    return quantile