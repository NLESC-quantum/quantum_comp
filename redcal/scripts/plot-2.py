# ~\~ language=Python filename=scripts/plot-2.py
# ~\~ begin <<README.md|scripts/plot-2.py>>[0]
# ~\~ begin <<README.md|imports>>[0]
import numpy as np
from numpy import matrix
from matplotlib import pyplot as plt
from pathlib import Path

def diag(x):
    return np.asmatrix(np.diag(x))
# ~\~ end
# ~\~ begin <<README.md|setup>>[0]
xpos = np.linspace(-2, 2, 5)[:,None]
# ~\~ end
# ~\~ begin <<README.md|setup>>[1]
n_ant = 5
# ~\~ end
# ~\~ begin <<README.md|setup>>[2]
l = np.r_[-0.5, 0.2, 0.7]
# ~\~ end
# ~\~ begin <<README.md|setup>>[3]
sigma = np.r_[0.8, 1, 0.4]
# ~\~ end
# ~\~ begin <<README.md|setup>>[4]
# g = 1 + 0.3 * (randn(Nant, 1) + 1i * randn(Nant, 1));
g  = 1 + 0.3 * (np.random.normal(size=n_ant) \
         + 1j * np.random.normal(size=n_ant))
# ~\~ end
# ~\~ begin <<README.md|setup>>[5]
freq = 150e6    # measurement frequency in MHz
c = 2.99792e8   # speed of light in m/s
A = np.matrix(np.exp(-(2 * np.pi * 1j * freq / c) * (xpos * l)))
# ~\~ end
# ~\~ begin <<README.md|setup>>[6]
R = diag(g) @ A @ diag(sigma) @ A.H @ diag(g).H
# ~\~ end
# ~\~ begin <<README.md|gain-magnitudes>>[0]
Mmag = np.matrix(
    [[1, 1, 0, 0, 0, 1, 0, 0],     # baseline type 1 (4 rows)
     [0, 1, 1, 0, 0, 1, 0, 0],
     [0, 0, 1, 1, 0, 1, 0, 0],
     [0, 0, 0, 1, 1, 1, 0, 0],
     [1, 0, 1, 0, 0, 0, 1, 0],     # baseline type 2 (3 rows)
     [0, 1, 0, 1, 0, 0, 1, 0],
     [0, 0, 1, 0, 1, 0, 1, 0],
     [1, 0, 0, 1, 0, 0, 0, 1],     # baseline type 3 (1 row)
     [0, 1, 0, 0, 1, 0, 0, 1],
     [1, 0, 0, 0, 0, 0, 0, 0]])    # magnitude constriant
# ~\~ end
# ~\~ begin <<README.md|gain-magnitudes>>[1]
# sel = np.c_[6, 12, 18, 24, 11, 17, 23, 16, 22].T - 1
sel = [i + j * n_ant + j             # indexing row-major
       for i in range(1, n_ant - 1)  # from first off-diagonal
                                     # not including the corner element
       for j in range(n_ant - i)]    # the first off-diagonal has (n_ant - 1) elements
                                     # and continuing, one less each time
# ~\~ end
# ~\~ begin <<README.md|gain-magnitudes>>[2]
theta = np.linalg.lstsq(Mmag, np.c_[np.log10(np.abs(R.flat[sel])), 0].T, rcond=None)
gmag = 10**np.asarray(theta[0])[:n_ant]
# ~\~ end
# ~\~ begin <<README.md|gain-phases>>[0]
Mph = np.array([
     [1, -1,  0,  0,  0,  1,  0,  0],   # baseline type 1 (4 rows)
     [0,  1, -1,  0,  0,  1,  0,  0],  
     [0,  0,  1, -1,  0,  1,  0,  0],  
     [0,  0,  0,  1, -1,  1,  0,  0],  
     [1,  0, -1,  0,  0,  0,  1,  0],   # baseline type 2 (3 rows)
     [0,  1,  0, -1,  0,  0,  1,  0],  
     [0,  0,  1,  0, -1,  0,  1,  0],  
     [1,  0,  0, -1,  0,  0,  0,  1],   # baseline type 3 (1 row)
     [0,  1,  0,  0, -1,  0,  0,  1],  
     [0,  0,  1,  0,  0,  0,  0,  0],   # phase constraint on first element
     np.r_[xpos.flat, 0, 0, 0]])             # phase gradient constraint
# ~\~ end
# ~\~ begin <<README.md|gain-phases>>[1]
theta = np.linalg.lstsq(Mph, np.c_[np.angle(R.flat[sel]), 0, 0].T, rcond=None)
gph = theta[0][:n_ant]
# ~\~ end

gph_true = np.c_[np.angle(g) - np.angle(g[2])]
theta = np.linalg.lstsq(xpos, gph_true - gph, rcond=None)
gph_true = gph_true - np.asarray(theta[0]) * xpos

fig, ax = plt.subplots(1, 1)
antennae = np.arange(1, n_ant+1)
ax.plot(antennae, gph_true, 'b-', label='true gain')
ax.plot(antennae, gph, 'ro', label='estimated gain')
ax.set_xlabel('antenna #id')
ax.set_ylabel('gain phases (rad)')
ax.legend()
Path("fig").mkdir(exist_ok=True)
fig.savefig("fig/plot-2.svg", bbox_inches='tight')
# ~\~ end
