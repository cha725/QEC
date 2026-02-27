import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from abc import ABC, abstractmethod
from itertools import product
from numpy.typing import NDArray

from binary_RREF import BinaryMatrix
from CSScodes import CSSStabiliserCode

class Codeword():
    """
    Represents a codeword in a linear code.
    """
    def __init__(self,
                 bits: list[int]):
        self.bits = bits

    def __len__(self) -> int:
        """ Returns length of the codeword, i.e. the number of columns. """
        return len(self.bits)

    def hamming_distance(self, other: "Codeword") -> int:
        """
        Returns the Hamming distance between this codeword and another.
        i.e. the number of entries where the codewords differ.
        """
        if len(self) != len(other):
            raise ValueError(f"{other} is not in the same code as {self}. Invalid length.")
        return sum(b1 ^ b2 for b1, b2 in zip(self.bits, other.bits))
    
    def __repr__(self):
        return f"Codeword({self.bits})"


class LinearCode(ABC):

    def __init__(self,
                 generators: list[list[int]] | None = None,
                 parity_checks: list[list[int]] | None = None):
        
        if generators is None and parity_checks is None:
            raise ValueError("Must provide either a generator or parity check matrix.")
        
        if generators is not None:
            self.generator_matrix = BinaryMatrix(generators).rowspace
        if parity_checks is not None:
            self.parity_check_matrix = BinaryMatrix(parity_checks).rowspace

        if generators is None:
            self.generator_matrix = self.parity_check_matrix.nullspace
        if parity_checks is None:
            self.parity_check_matrix = self.generator_matrix.nullspace

        self._validate_code()

        self.length: int = self.generator_matrix.num_cols
        self.rank: int = self.generator_matrix.rank
        self.rate: float = self.rank / self.length
        
        self.generators = self.generator_matrix.basis

        self.num_parity_checks = self.parity_check_matrix.rank
        self.parity_checks = self.parity_check_matrix.basis

    def _validate_code(self):
        """
        Check generator and parity check matrices are compatible.
        """
        if self.generator_matrix.num_cols != self.parity_check_matrix.num_cols:
            raise ValueError("Generator and parity check matrix must have the same number of columns")
        code_length = self.generator_matrix.num_cols
        if code_length != self.generator_matrix.rank + self.parity_check_matrix.rank:
            raise ValueError("")
        if not self.generator_matrix.is_perpendicular_to(self.parity_check_matrix):
            raise ValueError("Generator and parity check matrix must be perpendicular.")

    @cached_property
    def codewords(self) -> set[list[int]]:
        """
        WARNING: Computes ALL 2^k codewords.
        Returns a set of all codewords as lists of bits.
        """
        return self.generator_matrix.rowspace_vectors
    
    def choose_random_codeword(self) -> list[int]:
        """
        Returns a random codeword from the code as a list of bits.
        """
        coeffs = [random.choice([0,1]) for _ in range(self.rank)]
        return self.encode(coeffs)

    def encode(self, message: list[int]) -> list[int]:
        """
        Encode message using linear code.
        Returns uG where u is the message and G is the generator matrix.
        """
        if len(message) != self.rank:
            raise ValueError(f"Message must have length {self.rank}.")
        array = (np.array(message) @ self.generator_matrix.array) % 2
        return array.tolist()
    
        """
        Transmit codeword over a noisy channel with given bit flip probabilities.
        """
        if flip_probabilities is None:
            flip_probabilities = [0.1]*self.length
        received_message = []
        for bit, p in zip(codeword, flip_probabilities):
            if np.random.rand() < p:
                received_message.append(1 - bit)
            else:
                received_message.append(bit)
        return received_message
    
    def send_and_decode_message(self, codeword: list[int], flip_probabilities: list[float] | None = None, verbose: bool = False) -> list[int] | tuple[list[int],list[int],list[int]]:
        """
        Send a codeword and decode received message over a noisy channel.
        If verbose return encoded, received and decoded messages.
        Else return just decoded message.
        """
        encoded = self.encode(codeword)
        received = self.transmit_codeword(encoded, flip_probabilities)
        decoded = self.decode(received)
        if verbose:
            return (encoded, received, decoded)
        return decoded 

    def syndrome(self, vector: NDArray):
        """
        Computes syndrome of vector.
        Returns Hv^T where v is the vector and H is the parity check matrix.
        """
        if vector.shape != [1, self.code_length]:
            raise ValueError(f"Invalid vector size. Must be [1,{self.code_length}].")
        M = np.matmul(vector.T, self.parity_check_matrix.array, dtype=bool)
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
    Uses majority vote for decoding.

    Attribute:
        - codeword_length (int): Length of the codeword.
    """
    def __init__(self,
                 codeword_length: int):
        if (codeword_length % 2) != 1:
            raise ValueError(f"Repetition code must have odd codeword length. Got {codeword_length}.")
        super().__init__([[1]*codeword_length])


    def majority_vote_decoder(self, received_message: list[int]) -> list[int]:
        """
        Implements majority vote decoder
        """
        if len(received_message) != self.length:
            raise ValueError(f"Message must be of length {self.length}.")
        return [sum(received_message) % 2]*self.length


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
                 num_parity_check_eqns: int):
        self.num_parity_check_eqns = num_parity_check_eqns
        num_parity_check_cols = 2**num_parity_check_eqns-1

        parity_check_cols = []
        for val in range(1, num_parity_check_cols + 1):
            bits = [(val >> idx) & 1 for idx in range(num_parity_check_eqns)]
            parity_check_cols.insert(0, np.array(bits))
        pc = np.array(parity_check_cols).T
        super().__init__(parity_checks=pc.tolist())

    def decode(self, received: list[int]) -> list[int]:
        syndrome = self.syndrome(received)
        idx = int("".join(map(str, syndrome[::-1])), 2)
        corrected = received.copy()
        if idx != 0:
            corrected[idx - 1] ^= 1
        return corrected


class LDPC(LinearCode):
    """ 
    Represents a LDPC code.
    parity_checks: list of binary strings which represent the parity check equations.
    """
    def __init__(self,
                 parity_checks: list[list[int]],
                 bit_flip_probs: list[float] | None = None):
        super().__init__(parity_checks=parity_checks)
        self.parity_checks = self.parity_check_matrix.basis
        self.num_parity_checks = len(self.parity_checks)
        if bit_flip_probs is None:
            bit_flip_probs = [np.random.uniform(0.1,0.3) for _ in range(self.length)]
        if len(bit_flip_probs) != self.length:
            raise ValueError(f"Received {len(bit_flip_probs)} bit-flip probabilities. Need {self.length}.")
        self.bit_flip_probs = bit_flip_probs

    def graph(self) -> nx.Graph:
        G = nx.Graph()
        for bit_idx, bit_flip_prob in enumerate(self.bit_flip_probs):
            G.add_node(bit_idx, bipartite=0, bit_flip_prob=bit_flip_prob)
        G.add_nodes_from(range(self.length, self.num_parity_checks), bipartite=1)
        for pc_idx, parity_check in enumerate(self.parity_checks):
            for bit_idx, bit in enumerate(parity_check):
                if bit == 1:
                    G.add_edge(bit_idx, self.length + pc_idx)
        return G
    
    def draw_graph(self, title: str = "LDPC Bipartite Graph", bits_colour: str = "skyblue", pc_colour: str = "lightgreen"):
        pos = nx.bipartite_layout(self.graph(), nodes=range(self.length))
        nx.draw(self.graph(), pos, with_labels=True, node_color=[bits_colour]*self.length + [pc_colour]*self.num_parity_checks)
        plt.title(title)
        plt.show()

    
        
import random

class RandomLDPC(LDPC):
    """
    Create an LDPC code for a given number of bits and parity check equations.
    """
    def __init__(self,
                 num_bits: int,
                 num_parity_checks: int,
                 parity_check_weights: list[int]):
        self.num_bits = num_bits
        self.num_parity_checks = num_parity_checks
        self.parity_check_weights = parity_check_weights
        self.parity_checks = self._create_random_parity_checks()
        super().__init__(self.parity_checks)
    
    def _create_random_parity_checks(self):
        parity_check_matrix = np.zeros((self.num_parity_checks, self.num_bits), dtype=int)
        for row_idx in range(self.num_parity_checks):
            weight = np.random.choice(self.parity_check_weights)
            col_idx = np.random.choice(self.num_bits, size=weight, replace=False)
            parity_check_matrix[row_idx, col_idx] = 1
        return parity_check_matrix.tolist()

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

    