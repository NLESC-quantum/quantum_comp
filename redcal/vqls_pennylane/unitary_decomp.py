import numpy as np
import scipy.linalg as spla

def unitary_decomposition(X):
    """ Unitary decomposition

    Decomposes the complex normalized matrix X into four unitary matrices
    """
    def get_real(x):
        """ Get the real part of x"""
        return 0.5 * (x + np.conjugate(x))

    def get_imag(x):
        """ Get the imaginary part of x """
        return 0.5/(1j) * (x - np.conjugate(x))

    def aux(x):
        """ Auxiliary function

        Performs a matrix operation that we'll need later
        """
        I = np.eye(len(x))
        return 1j*spla.sqrtm(I - x**2)

    # Normalize
    norm = np.linalg.norm(X)
    X_n = X / norm

    # Apply the algorithm as described in
    # https://math.stackexchange.com/questions/1710247/every-matrix-can-be-written-as-a-sum-of-unitary-matrices/1710390#1710390
    B = get_real(X_n)
    C = get_imag(X_n)

    ## Get the matrices
    UB = B + aux(B)
    UC = C + aux(C)
    VB = B - aux(B)
    VC = C - aux(C)

    ## Get the coefficients
    cb = norm * 0.5
    cc = cb * 1j

    ## Return
    return [cb,cb,cc,cc], [UB, VB, UC, VC]


def unitary_recomposition(decomposed):
    """ Rebuilds the original matrix from the decomposed one """
    recomp = decomposed[0][0] * decomposed[0][1]
    for c,m in decomposed[1:]:
        recomp += c * m
    return recomp


def get_offdiag(m):
    return m - np.diag(np.diag(m))

def get_holes(m):
    return np.diag(np.array(np.sum(m!=0,1)==0).flatten()).astype('float')

def process_offdiag(m):

    unique_values = np.unique(np.asarray(m))
    z = np.array([0])
    unique_values = np.setdiff1d(unique_values,z)

    vals = []
    mats = []
    holes = []
    for uv in unique_values:
        tmp = np.zeros_like(m)
        tmp[m==uv]=uv
        h = get_holes(tmp)
        holes.append(uv*h)
        mats.append(tmp+uv*h)
        vals.append(uv)

    return mats, holes, vals

def manual_decomp(mat, nant, nbaselines):


    main_diag = np.diag(np.diag(mat.T@mat)).astype('float')
    eye = np.eye(len(nbaselines)-1)
    zeros = np.zeros((nant, len(nbaselines)-1))

    mats, ids = [], []
    incr = 0
    for nb in nbaselines[:-1]:
        mats.append(mat[incr:incr+nb,:nant])
        ids.append(mat[incr:incr+nb,nant:])
        incr += nb

    off_diag_matrix = []
    alpha = []
    for n in range(len(nbaselines)-1):

        mm = mats[n].T @ mats[n]
        off_diag_mm = get_offdiag(mm)
        x,y,z = process_offdiag(off_diag_mm)

        for (od, h, v) in zip(x,y,z):
            alpha.append(v)
            off_diag_matrix.append(np.block([[od, zeros],[zeros.T, v*eye]]))
            if np.sum(h!=0)>=1:
                main_diag -= np.block([[h, zeros],[zeros.T, v*eye]])


    for n in range(len(nbaselines)-1):
        mi = mats[n].T @ ids[n]
        tmp = np.block([[np.zeros((nant,nant)), mi],[mi.T,np.zeros((len(nbaselines)-1,len(nbaselines)-1))]])
        x,y,z = process_offdiag(tmp)

        for (od, h, v) in zip(x,y,z):
            alpha.append(v)
            off_diag_matrix.append(od)
            main_diag -= h

    return main_diag, off_diag_matrix, alpha
