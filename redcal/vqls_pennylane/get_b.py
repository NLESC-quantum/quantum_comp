from qiskit import QuantumCircuit
from qiskit import execute, Aer
import math
import numpy as np


def get_b_unitmatrix(desired_vector, nqbit, decimals=3):


    # desired_vector /= np.linalg.norm(desired_vector)

    qc = QuantumCircuit(nqbit) #circuit with 1 qubit
    qc.initialize(desired_vector, list(range(nqbit))) #0 in the index of the qubit

    # simulator=Aer.get_backend('statevector_simulator')
    # job = execute(qc, simulator)
    # qc_state = job.result().get_statevector(qc)

    usimulator=Aer.get_backend('unitary_simulator')

    job = execute(qc, usimulator)
    umatrix = job.result().get_unitary(qc,decimals=decimals) #decimals is not necessary
    return umatrix


n = 3
b =  np.random.rand(2**n)
b /= np.linalg.norm(b)
print(b)
u = get_b_unitmatrix(b,n)
print(u)
