from dwave.system import DWaveSampler , EmbeddingComposite
import neal
from dimod import ExactSolver
import math
import random
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from qubo_redcal_utils import get_antnennas_response
from create_qubo_matrix import SolutionVector, create_qubo_matrix
from create_qubo_matrix import RealUnitQbitEncoding
import dwave
import dwave.inspector

norm = 1

n_ant = 5
xpos = np.linspace(-2, 2, n_ant)[:,None]
b = get_antnennas_response(xpos)
# b[-1] = 1.
b /= norm

# print(b)




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
     [1, 0, 0, 0, 0, 0, 0, 0]])    # magnitude constraint




# Mmag = np.array([[3,1],[-1,2]])
# b = np.array([[-1.],[5.]])

npsol = np.linalg.lstsq(Mmag,b, rcond=None)
npsol = np.asarray(npsol[0]).flatten()*norm
print(npsol)



sol = SolutionVector(size=Mmag.shape[1], nqbit=11, encoding=RealUnitQbitEncoding)
x = sol.create_polynom_vector()
qubo_dict = create_qubo_matrix(Mmag, b, x, prec=None)


sampler = neal.SimulatedAnnealingSampler()
sampleset = sampler.sample_qubo(qubo_dict,num_reads=1000)
lowest_sol = sampleset.lowest()
sol_num = sol.decode_solution(lowest_sol.record[0][0])*norm
print(sol_num)



plt.scatter(npsol, sol_num)
plt.plot([-1,1],[-1,1],'--',c='gray')
plt.show()


exit()

sampler = EmbeddingComposite(DWaveSampler(solver={'qpu':True}))
sampleset = sampler.sample_qubo(qubo_dict,num_reads=1000,chain_strength=1000)
lowest_sol = sampleset.lowest()
sol_num = sol.decode_solution(lowest_sol.record[0][0])

plt.scatter(npsol, sol_num)
plt.plot([-1,1],[-1,1],'--',c='gray')
plt.show()


dwave.inspector.show(sampleset)