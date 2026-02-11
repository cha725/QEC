import numpy as np

from collections.abc import Sequence
from numpy.typing import NDArray

from binary_RREF import BinaryMatrix
from CSScodes import CSSStabiliserCode

class Codeword():
    def __init__(self,
                 vector: BinaryMatrix):
        if vector.shape[0] != 1:
            raise ValueError(f"Codeword must be a row vector. Got {vector.shape[0]} rows.")
        self.vector = vector

    def __len__(self) -> int:
        """ Returns length of the codeword, i.e. the number of columns. """
        return self.vector.shape[1]


class LinearCode():
    def __init__(self,
                 generator_matrix: BinaryMatrix | None = None,
                 parity_check_matrix: BinaryMatrix | None = None):
        if generator_matrix is None and parity_check_matrix is None:
            raise ValueError("Must provide either a generator or parity check matrix.")
        if generator_matrix is not None:
            self.generator_matrix = generator_matrix
            self.n = generator_matrix.shape[1]
            self.rank = generator_matrix.rank
            self.parity_check_matrix = generator_matrix.nullspace
        if parity_check_matrix is not None:
            self.parity_check_matrix = parity_check_matrix
            self.n = parity_check_matrix.shape[1]
            self.rank = self.n - parity_check_matrix.rank
            self.generator_matrix = parity_check_matrix.nullspace
        
        self.validate_code()
        self.rate: float = self.n / self.rank

    def validate_code(self):
        """
        Check generator and parity check matrices are compatible.
        """
        if self.rank < self.generator_matrix.rank:
            raise ValueError("Rank of parity check matrix is too small.")
        if self.rank > self.generator_matrix.rank:
            raise ValueError("Rank of parity check matrix is too large.")
        M = np.matmul(self.generator_matrix.array, self.parity_check_matrix.array.T)
        if not np.all(M % 2 == 0):
            raise ValueError("Generator and parity check matrices are not orthogonal.")

    def encode(self, message: NDArray):
        """
        Encode message using linear code.
        Returns uG where u is the message and G is the generator matrix.
        """
        if message.shape[1] != self.rank:
            raise ValueError(f"Invalid message length. Must have {self.rank} columns.")
        M = np.matmul(message, self.generator_matrix.array, dtype=bool)
        return np.array(M, dtype=int)
    
    def syndrome(self, vec: NDArray):
        """
        Computes syndrome of vector.
        Returns Hv^T where v is the vector and H is the parity check matrix.
        """
        if vec.shape != [1, self.n]:
            raise ValueError(f"Invalid vector size. Must be [1,{self.n}].")
        M = np.matmul(vec.T, self.parity_check_matrix.array, dtype=bool)
        return np.array(M, dtype=int)

    def css_code_from_linear(self):
        """
        Create corresponding CSS code.
        """
        G = self.generator_matrix.array
        z_vecs = [list(G[idx,:]) for idx in range(G.shape[0])]
        H = self.parity_check_matrix.array
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
        self.generator_matrix = self.dual_code.parity_check_matrix
        super().__init__(BinaryMatrix(self.generator_matrix.array))

if __name__ == "__main__":

    class Examples:
        def __init__(self,
                     examples: Sequence[LinearCode]):
            self.examples = examples

        def print(self):
            for code in self.examples:
                print(f"=== The code type is: {code.__class__.__name__} === \n")
                print(f"[n, k] = [{code.n}, {code.rank}] \n")
                print(f"Generator matrix: \n {code.generator_matrix} \n")
                print(f"Parity check matrix: \n {code.parity_check_matrix} \n")
                print(f"=== Induced CSS code === \n")
                css_code = code.css_code_from_linear()
                if css_code.all_commute:
                    print("All stabilisers commute. \n")
                print("List of stabilisers:\n")
                stab_generators = css_code.stab_generators
                for generator in stab_generators:
                    print(generator)
                print()

    examples = Examples([RepetitionCode(4),
                         HammingCode(3)])
    
    examples.print()

    