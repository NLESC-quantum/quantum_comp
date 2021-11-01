import numpy as np
from qiskit import QuantumCircuit


def BitsToIntAFast(bits, invert_bits_order=True):
    n = len(bits)  # number of columns is needed, not bits.size
    # -1 reverses array of powers of 2 of same length as bits
    a = 2**np.arange(n)
    if invert_bits_order:
        a = a[::-1]
    return bits @ a  # this matmult is the key line of code


def qft_dagger(n):
    """n-qubit QFTdagger the first n qubits in circ"""
    qc = QuantumCircuit(n)
    # Don't forget the Swaps!
    for qubit in range(n//2):
        qc.swap(qubit, n-qubit-1)
    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi/float(2**(j-m)), m, j)
        qc.h(j)
    qc.name = "QFT†"
    return qc

def tobin(x, length):
    xbin = bin(x).replace('0b','')
    while len(xbin)<length:
        xbin = '0'+xbin
    return xbin