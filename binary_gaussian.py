import numpy as np
from numpy.typing import NDArray

class PermutationMatrix:
    """
    Take in a list of pairs of indices and produce a matrix that performs each of those swaps.
    
    Notes:
        - Precomposition by a permutation matrix swaps rows.
        - Postcomposition by a permutation matrix swaps columns.
    
    Attributes:
        - row_swap (bool): If True, swaps rows; otherwise swaps columns
        - pairs_to_swap (list[list[int]]): List of pairs of indices
        - matrix_dim: Dimension of the square matrix
    """
    
    def __init__(self,
                 row_swap: bool = True,
                 pairs_to_swap: list[list[int]] | None = None,
                 matrix_dim: int = 0):
        
        if pairs_to_swap is not None:
            for i, j in pairs_to_swap:
                if not (0 <= i < matrix_dim and 0 <= j < matrix_dim):
                    raise ValueError(f"Invalid indices. Both entries must be in [0, {matrix_dim-1}]. Got {i, j}.")
        
        self.row_swap = row_swap
        self.pairs_to_swap = pairs_to_swap or []
        self.matrix_dim = matrix_dim
        self.np_array = self._create_permutation()

    def _create_permutation(self):
        """
        Create permutation matrix.
        """
        P = np.identity(self.matrix_dim)
        for i, j in self.pairs_to_swap:
            if self.row_swap:
                P[[i, j], :] = P[[j, i], :]
            else:
                P[:, [i, j]] = P[:, [j, i]]
        return P
    
def row_swap(M: NDArray, i: int, j: int):
    """
    Swap two rows in a matrix.

    Parameters:
        - m (NDArray): The matrix to switch the rows in.
        - i (int): The index of the first row to be swapped.
        - j (int): The index of the second row to be swapped.
    """
    M[[i,j], :] = M[[j,i], :]
    return M




if __name__ == "__main__":

    P = PermutationMatrix(True, [[1,2]], 3)
    P = P.np_array
    print(P)
    
    M = np.array([[1,2,3],[4,5,6]])
    print(row_swap(M,0,1))
    

