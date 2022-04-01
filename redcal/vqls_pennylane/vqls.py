import numpy as np
from numpy import matrix
from matplotlib import pyplot as plt
from pathlib import Path
from unitary_decomp import unitary_decomposition, manual_decomp
from qiskit.quantum_info.operators import Operator

def diag(x):
    return np.asmatrix(np.diag(x))

xpos = np.linspace(-2, 2, 5)[:,None]
n_ant = 5
l = np.r_[-0.5, 0.2, 0.7]
sigma = np.r_[0.8, 1, 0.4]


# g = 1 + 0.3 * (randn(Nant, 1) + 1i * randn(Nant, 1));
g  = 1 + 0.3 * (np.random.normal(size=n_ant) \
         + 1j * np.random.normal(size=n_ant))


freq = 150e6    # measurement frequency in MHz
c = 2.99792e8   # speed of light in m/s
A = np.matrix(np.exp(-(2 * np.pi * 1j * freq / c) * (xpos * l)))

R = diag(g) @ A @ diag(sigma) @ A.H @ diag(g).H

Mmag = np.matrix(
    [[1, 1, 0, 0, 0, 1, 0, 0],     # baseline type 1 (4 rows)
     [0, 1, 1, 0, 0, 1, 0, 0],
     [0, 0, 1, 1, 0, 1, 0, 0],
     [0, 0, 0, 1, 1, 1, 0, 0],
     [1, 0, 1, 0, 0, 0, 1, 0],     # baseline type 2 (3 rows)
     [0, 1, 0, 1, 0, 0, 1, 0],
     [0, 0, 1, 0, 1, 0, 1, 0],
     [1, 0, 0, 1, 0, 0, 0, 1],     # baseline type 3 (2 row)
     [0, 1, 0, 0, 1, 0, 0, 1],
     [1, 0, 0, 0, 0, 0, 0, 0]])    # magnitude constriant


# sel = np.c_[6, 12, 18, 24, 11, 17, 23, 16, 22].T - 1
sel = [i + j * n_ant + j             # indexing row-major
       for i in range(1, n_ant - 1)  # from first off-diagonal
                                     # not including the corner element
       for j in range(n_ant - i)]    # the first off-diagonal has (n_ant - 1) elements
                                     # and continuing, one less each time

b = np.c_[np.log10(np.abs(R.flat[sel])),0].T

# theta = np.linalg.lstsq(Mmag, b, rcond=None)
# gmag = 10**np.asarray(theta[0])[:n_ant]

theta = np.linalg.solve(Mmag.T@Mmag, Mmag.T @ b)
gmag = 10**np.asarray(theta)[:n_ant]
print((theta).shape)



# Normalise true gain values to match constraint that the gain of the first
gmag_true = abs(g) / abs(g.flat[0])
fig, ax = plt.subplots(1, 1)
antennae = np.arange(1, n_ant+1)
ax.plot(antennae, gmag_true, 'b-', label='true gain')
ax.plot(antennae, gmag, 'ro', label='estimated gain')
ax.set_xlabel('antenna #id')
ax.set_ylabel('gain magnitude')
ax.legend()
# plt.show()


a = Mmag.T@Mmag
mats = unitary_decomposition(a)

# acpy = np.zeros_like(a).astype('complex')
# for m in mats:
    # acpy += m[0][0]*m[1]


# diag_mat, offdiag_mat, coefs = manual_decomp(Mmag, 5, [4, 3, 2, 1])
print(mats[0])
op = Operator(mats[0][1])
op.label = 'A'

