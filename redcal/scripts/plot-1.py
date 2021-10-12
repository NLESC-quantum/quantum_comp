# ~\~ language=Python filename=scripts/plot-1.py
# ~\~ begin <<README.md|scripts/plot-1.py>>[0]
# ~\~ begin <<README.md|imports>>[0]
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
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
A = np.exp(-(2 * np.pi * 1j * freq / c) * (xpos * l))
# ~\~ end
# ~\~ begin <<README.md|setup>>[6]
R = np.diag(g) @ A @ np.diag(sigma) @ A.T @ np.diag(g).T
# ~\~ end
# ~\~ begin <<README.md|calibration>>[0]
Mmag = np.array(
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
# ~\~ begin <<README.md|calibration>>[1]
# sel = np.c_[6, 12, 18, 24, 11, 17, 23, 16, 22].T - 1
sel = [i + j * n_ant + j             # indexing row-major
       for i in range(1, n_ant - 1)  # from first off-diagonal
                                     # not including the corner element
       for j in range(n_ant - i)]    # the first off-diagonal has (n_ant - 1) elements
                                     # and continuing, one less each time
# ~\~ end
# ~\~ begin <<README.md|calibration>>[2]
theta = np.linalg.lstsq(Mmag, np.r_[np.log10(np.abs(R.flat[sel])), 0])
gmag = 10**theta[0][:n_ant]
# ~\~ end

gmag_true = abs(g) / abs(g.flat[0])
fig, ax = plt.subplots(1, 1)
antennae = np.arange(1, n_ant+1)
ax.plot(antennae, gmag_true, 'b-', label='true gain')
ax.plot(antennae, gmag, 'ro', label='estimated gain')
ax.set_xlabel('antenna #id')
ax.set_ylabel('gain magnitude')
Path("fig").mkdir(exist_ok=True)
fig.savefig("fig/plot-1.svg", bbox_inches='tight')
# ~\~ end