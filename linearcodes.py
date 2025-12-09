from collections.abc import Sequence
import numpy as np

from binary_RREF import BinaryMatrix
from CSScodes import CSSStabiliserCode

class LinearCode():
    def __init__(self,
                 generators: BinaryMatrix | None = None):
        if generators is None:
            raise ValueError("Must provide at least one generator.")
        self.generators = generators
        self.n = generators.matrix.shape[1]
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

class RepetitionCode(LinearCode):
    """
    Represents a repetition code as a subclass of LinearCode.

    A repetition code is the code with two codewords 0 and 1.

    Attribute:
        - codeword_length (int): Length of the codeword.
    """
    def __init__(self,
                 codeword_length: int):
        B = BinaryMatrix(np.array([[1 for _ in range(codeword_length)]]))
        super().__init__(B)


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

    class Examples:
        def __init__(self,
                     examples: Sequence[LinearCode]):
            self.examples = examples

        def print(self):
            for code in self.examples:
                print(type(code))
                print(f"Generator matrix: \n {code.generator_matrix}")
                print(f"[n, k] = [{code.n}, {code.rank}]")
                print(f"Parity check matrix: \n {code.parity_check}")
                css_code = code.css_code_from_linear()
                print(css_code.all_commute)
                print(css_code.stab_generators)

    examples = Examples([RepetitionCode(4),
                         HammingCode(3)])
    
    examples.print()

    