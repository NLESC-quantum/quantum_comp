from qiskit import QuantumCircuit
from qiskit import execute, Aer
import math
import numpy as np


def get_b_unitmatrix(desired_vector, nqbit, decimals=6):
    """Get unitary matrix that generates |b> from |0>

    We do that by using Qiskit to create a circuit,
    initialize the state of the circuit to |b>
    and extract the matrix of the circuit.
    There might be better ways of doing this."""

    # desired_vector /= np.linalg.norm(desired_vector)
    norm = np.linalg.norm(desired_vector)
    desired_vector = np.asarray(desired_vector).flatten()
    desired_vector /= norm


    qc = QuantumCircuit(nqbit) #circuit with 1 qubit
    qc.initialize(desired_vector, list(range(nqbit))) #0 in the index of the qubit
    usimulator=Aer.get_backend('unitary_simulator')
    job = execute(qc, usimulator)
    umatrix = job.result().get_unitary(qc,decimals=decimals) #decimals is not necessary
    return norm, umatrix

