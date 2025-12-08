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
        self.gen_matrix = self.generators.generator_matrix
        self.rank = self.generators.rank
        self.minimal_generators = [self.gen_matrix[idx,:] for idx in range(self.rank)]
        self.parity_check = self.generators.parity_check_matrix

    def css_code_from_linear(self):
        """
        Create corresponding CSS code.
        """
        G = self.gen_matrix
        z_vecs = [list(G[idx,:]) for idx in range(G.shape[0])]
        H = self.parity_check
        x_vecs = [list(H[idx,:]) for idx in range(H.shape[0])]
        return CSSStabiliserCode(z_vecs=z_vecs, x_vecs=x_vecs)


if __name__ == "__main__":

    ### Repetition code ###
    B = BinaryMatrix(np.array([[1,1,1]]))
    LC = LinearCode(B)
    code = LC.css_code_from_linear()
    print(code.all_commute)
    print(code.stab_generators)