import numpy as np
from numpy import matrix
from matplotlib import pyplot as plt
from pathlib import Path


def diag(x):
    return np.asmatrix(np.diag(x))

def get_antnennas_response(xpos):

    n_ant = len(xpos)
    freq = 150e6    # measurement frequency in MHz
    c = 2.99792e8   # speed of light in m/s

    l = np.r_[-0.5, 0.2, 0.7]
    sigma = np.r_[0.8, 1, 0.4]

    g  = 1 + 0.3 * (np.random.normal(size=n_ant) \
       + 1j * np.random.normal(size=n_ant))

    A = np.matrix(np.exp(-(2 * np.pi * 1j * freq / c) * (xpos * l)))

    R = diag(g) @ A @ diag(sigma) @ A.H @ diag(g).H

    # sel = np.c_[6, 12, 18, 24, 11, 17, 23, 16, 22].T - 1
    sel = [i + j * n_ant + j             # indexing row-major
        for i in range(1, n_ant - 1)  # from first off-diagonal
                                        # not including the corner element
        for j in range(n_ant - i)]    # the first off-diagonal has (n_ant - 1) elements
                                        # and continuing, one less each time

    b = np.c_[np.log10(np.abs(R.flat[sel])),0].T

    return b
