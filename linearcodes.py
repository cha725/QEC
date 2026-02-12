import numpy as np

from abc import ABC, abstractmethod
from collections.abc import Sequence
from numpy.typing import NDArray

from binary_RREF import BinaryMatrix
from CSScodes import CSSStabiliserCode

class Codeword():
    def __init__(self,
                 vector: list[int]):
        self.vector = vector

    def __len__(self) -> int:
        """ Returns length of the codeword, i.e. the number of columns. """
        return len(self.vector)

    def hamming_distance(self, other: "Codeword") -> int:
        """
        Returns the Hamming distance between this codeword and another.
        i.e. the number of entries where the codewords differ.
        """
        if len(self) != len(other):
            raise ValueError(f"{other} is not a codeword in the same code as {self}. Invalid length.")
        return sum(s_entry ^ o_entry for s_entry in self.vector for o_entry in other.vector)


class LinearCode(ABC):
    def __init__(self,
                 generators: list[list[int]] | None = None,
                 parity_checks: list[list[int]] | None = None):
        # There is an issue here, too many uses of the word generators
        if generators is None and parity_checks is None:
            raise ValueError("Must provide either a generator or parity check matrix.")
        if generators is not None:
            self.generator_matrix = BinaryMatrix(generators).rowspan_matrix()
            self.n = self.generator_matrix.shape[1]
            self.rank = self.generator_matrix.rank
            self.parity_check_matrix = self.generator_matrix.nullspace
        if parity_checks is not None:
            self.parity_check_matrix = BinaryMatrix(parity_checks).rowspan_matrix()
            self.n = self.parity_check_matrix.shape[1]
            self.rank = self.n - self.parity_check_matrix.rank
            self.generator_matrix = self.parity_check_matrix.nullspace
        
        self.validate_code()
        self.rate: float = self.n / self.rank
        self.length = self.generator_matrix.shape[1]
        self.elements = self.generator_matrix.rowspan_elements()
        self.codewords = [Codeword(elt) for elt in self.elements]
        self.basis = self.generator_matrix.basis


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

    def message_space(self) -> list[list[int]]:
        """
        Return the list of valid messages to be encoded.
        """
        return [list(entry) for entry in product([0,1], repeat=self.rank)]

    def encode(self, message: list[int]) -> list[int]:
        """
        Encode message using linear code.
        Returns uG where u is the message and G is the generator matrix.
        """
        if len(message) != self.rank:
            raise ValueError(f"Invalid message length. Must have {self.rank} columns.")
        mat_message = np.array(message, dtype=bool)
        bool_generator_mat = self.generator_matrix.array.astype(bool)
        bool_encoded = np.matmul(mat_message, bool_generator_mat, dtype=bool)
        encoded = bool_encoded.astype(int)
        return encoded.tolist()
    
    @abstractmethod
    def decode(self, message: list[int]) -> list[int]:
        """ Method to decode a received message. To be implemented in subclasses. """
        pass
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

    