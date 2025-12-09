import numpy as np
from numpy.typing import NDArray
from typing import Literal
bit = Literal[0,1]

from binary_RREF import BinaryMatrix
from CSScodes import CSSStabiliserCode

class LinearCode():
    def __init__(self,
                 generators: BinaryMatrix | None = None):
        if generators is None:
            raise ValueError("Must provide at least one generator.")
        self.generators = generators
        self.generator_matrix = self.generators.generator_matrix
        self.rank = self.generators.rank
        self.minimal_generators = [self.generator_matrix[idx,:] for idx in range(self.rank)]
        self.parity_check = self.generators.parity_check_matrix

    def css_code_from_linear(self):
        """
        Create corresponding CSS code.
        """
        G = self.generator_matrix
        z_vecs = [list(G[idx,:]) for idx in range(G.shape[0])]
        H = self.parity_check
        x_vecs = [list(H[idx,:]) for idx in range(H.shape[0])]
        return CSSStabiliserCode(z_vecs=z_vecs, x_vecs=x_vecs)

class HammingCode(LinearCode):
    """
    Represents a Hamming code as a subclass of LinearCode.

    A Hamming code is a code with parity check matrix defined
    by the binary expansion of 2^n-1 to 1. The parity check matrix
    is of the form:
            [ 2^n-1 | 2^n - 2 | ... | 2 | 1 ]
            where each column is the binary expansion
            of the corresponding integer.

    Attributes:
        - num_pc_rows (int): number of rows in the parity check matrix.
    """
    def __init__(self,
                 num_pc_rows: int):
        self.num_pc_rows = num_pc_rows
        self.num_pc_cols = 2**num_pc_rows-1

        parity_check_cols = []
        for num in range(1, self.num_pc_cols + 1):
            bin_vec = [(num >> idx) & 1 for idx in range(self.num_pc_rows)]
            parity_check_cols.insert(0, np.array(bin_vec))
        H = BinaryMatrix(np.array(parity_check_cols).T)
        self.dual_code = LinearCode(H)
        self.parity_check = self.dual_code.generator_matrix
        self.generator_matrix = self.dual_code.parity_check
        super().__init__(BinaryMatrix(self.generator_matrix))

if __name__ == "__main__":

    ### Repetition code ###
    B = BinaryMatrix(np.array([[1,1,1]]))
    LC = LinearCode(B)
    code = LC.css_code_from_linear()
    print(code.all_commute)
    print(code.stab_generators)

    ### Hamming code [7,4,1] ###4
    H = HammingCode(3)
    print("generator matrix", H.generator_matrix)
    print("parity check", H.parity_check)
    css_code = H.css_code_from_linear()
    print(css_code.all_commute)
    print(css_code.stab_generators)