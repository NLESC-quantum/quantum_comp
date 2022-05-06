from sympy import Symbol
from sympy.matrices import Matrix
import numpy as np


class BaseQbitEncoding(object):

    def __init__(self, nqbit, var_base_name):
        """Encode a  single real number in a

        Args:
            nqbit (int): number of qbit required in the expansion
            var_base_name (str): base names of the different qbits
            only_positive (bool, optional): Defaults to False.
        """
        self.nqbit = nqbit
        self.var_base_name = var_base_name
        self.variables = self.create_variable()


    def create_variable(self):
        """Create all the variabes/qbits required for the expansion

        Returns:
            list: list of Symbol
        """
        variables = []
        for i in range(self.nqbit):
            variables.append(Symbol(self.var_base_name + '_%02d' %(i+1)))
        return variables

class RealQbitEncoding(BaseQbitEncoding):

    def __init__(self, nqbit, var_base_name):
        super().__init__(nqbit, var_base_name)
        self.base_exponent = 0

    def create_polynom(self):
        """
        Create the polynoms of the expansion

        Returns:
            sympy expression
        """
        out = 0.0
        for i in range(self.nqbit//2):
            out += 2**(i-self.base_exponent) * self.variables[i]
            out -= 2**(i-self.base_exponent) * self.variables[self.nqbit//2+i]
        return out

    def decode_polynom(self, data):
        out = 0.0
        for i in range(self.nqbit//2):
            out += 2**(i-self.base_exponent) * data[i]
            out -= 2**(i-self.base_exponent) * data[self.nqbit//2+i]
        return out

class RealUnitQbitEncoding(BaseQbitEncoding):

    def __init__(self, nqbit, var_base_name):
        super().__init__(nqbit, var_base_name)
        self.base_exponent = 0

    def create_polynom(self):
        """
        Create the polynoms of the expansion

        Returns:
            sympy expression
        """
        out = 0.0
        self.int_max = 0.0
        for i in range(self.nqbit//2):
            self.int_max += 2**(i-self.base_exponent)
        for i in range(self.nqbit//2):
            out += 2**(i-self.base_exponent)/self.int_max * self.variables[i]
            out -= 2**(i-self.base_exponent)/self.int_max * self.variables[self.nqbit//2+i]
        return out

    def decode_polynom(self, data):
        out = 0.0
        for i in range(self.nqbit//2):
            out += 2**(i-self.base_exponent)/self.int_max * data[i]
            out -= 2**(i-self.base_exponent)/self.int_max * data[self.nqbit//2+i]
        return out

class PositiveQbitEncoding(BaseQbitEncoding):

    def __init__(self, nqbit, var_base_name):
        super().__init__(nqbit, var_base_name)

    def create_polynom(self):
        """
        Create the polynoms of the expansion

        Returns:
            sympy expression
        """
        out = 0.0
        for i in range(self.nqbit):
            out += 2**i * self.variables[i]
        return out

    def decode_polynom(self, data):
        out = 0.0
        for i in range(self.nqbit//2):
            out += 2**i * data[i]
            out -= 2**i * data[self.nqbit//2+i]
        return out

class SolutionVector(object):

    def __init__(self, size, nqbit, encoding, base_name = 'x'):
        """Encode the solution vector in a list of RealEncoded

        Args:
            size (int): number of unknonws in the vector (i.e. size of the system)
            nqbit (int): number of qbit required per unkown
            base_name (str, optional): base name of the unknowns Defaults to 'x'.
            only_positive (bool, optional):  Defaults to False.
        """
        self.size = size
        self.nqbit = nqbit
        self.base_name = base_name
        self.encoding = encoding
        self.encoded_reals = self.create_encoding()

    def create_encoding(self):
        """Create the eocnding for all the unknowns


        Returns:
            list[RealEncoded]:
        """
        encoded_reals = []
        for i in range(self.size):
            var_base_name = self.base_name + str(i+1)
            encoded_reals.append(self.encoding(self.nqbit, var_base_name))

        return encoded_reals

    def create_polynom_vector(self):
        """Create the list of polynom epxressions

        Returns:
            sympy.Matrix: matrix of polynomial expressions
        """
        pl = []
        for real in self.encoded_reals:
            pl.append(real.create_polynom())

        return Matrix(pl)

    def decode_solution(self, data):

        sol = []
        for i, real in enumerate(self.encoded_reals):
            local_data = data[i*self.nqbit:(i+1)*self.nqbit]
            sol.append(real.decode_polynom(local_data))
        return np.array(sol)


def create_qubo_matrix(Anp, bnp, x):
    """Create the QUBO dictionary requried by dwave solvers
       to solve the linear system

       A x = b

    Args:
        Anp (np.array): matrix of the linear system
        bnp (np.array): righ hand side of the linear system
        x (sympy.Matrix): unknown

    Returns:
        _type_: _description_
    """
    A = Matrix(Anp)
    b = Matrix(bnp)

    polynom = x.T @ A.T @ A @ x - x.T @ A.T @ b - b.T@ A @ x + b.T @ b
    polynom = polynom[0]
    polynom = polynom.expand()
    polynom = polynom.as_ordered_terms()

    out = dict()

    for term in polynom:
        m = term.args
        if len(m) == 0:
            continue

        if len(m) == 2:
            varname = str(m[1])
            varname = varname.split("**")[0]
            key = (varname , varname)

        elif len(m) == 3:
            key = (str(m[1]),str(m[2]))

        if key not in out:
            out[key] = 0.0

        out[key] += m[0]

    return out

if __name__ == "__main__":

    from dwave.system import DWaveSampler , EmbeddingComposite
    import neal
    from dimod import ExactSolver

    A = np.array([[3,1],[-1,2]])
    b = np.array([[-1],[5]])

    sol = SolutionVector(size=2, nqbit=4)
    x = sol.create_polynom_vector()

    qubo_dict = create_qubo_matrix(A, b, x)

    # sampler = EmbeddingComposite(DWaveSampler(solver={'qpu': True}))
    sampler = neal.SimulatedAnnealingSampler()
    # sampler = ExactSolver()

    sampleset = sampler.sample_qubo(qubo_dict)
    lowest_sol = sampleset.lowest()
    sol_num = sol.decode_solution(lowest_sol.record[0][0])
    print(np.linalg.solve(A,b))
    print(sol_num)
    print(lowest_sol)