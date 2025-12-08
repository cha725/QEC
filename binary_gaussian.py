import numpy as np
from numpy.typing import NDArray

    
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
    

