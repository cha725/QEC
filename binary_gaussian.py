import numpy as np
from numpy.typing import NDArray

def row_swap(M: NDArray[np.bool_], i: int, j: int) -> NDArray[np.bool_]:
    """
    Swap two rows in a matrix.

    Parameters:
        - m (NDArray): The matrix to switch the rows in.
        - i (int): The index of the first row to be swapped.
        - j (int): The index of the second row to be swapped.
    """
    M = M.copy()
    M[[i,j], :] = M[[j,i], :]
    return M

def add_rows(M: NDArray[np.bool_], i: int, j: int) -> NDArray[np.bool_]:
    """
    Add two rows in the matrix.
    Note: Add row j to row i. 
    The new row i is i+j and row j remains untouched.

    Parameters:
        - M (NDArray): The matrix where we add to rows.
        - i (int): The index of the row to add to. Row i will be row i + row j.
        - j (int): The index of the row to add to row i.
    """
    M = M.copy()
    M[[i],:] ^= M[[j],:]
    return M

def nonzero_in_col(M: NDArray[np.bool_], i: int) -> list[int]:
    """
    Return a list of row indices of a matrix that are nonzero in a specific column.

    Parameters:
        - M (NDArray): The matrix to check.
        - i (int): The column index to check.
    """
    col = M[:, i]
    return [idx for idx, x in enumerate(col) if x]




if __name__ == "__main__":
    
    M = np.array([[1,0,1],[1,1,0]])
    print(row_swap(M,0,1))

    N = np.array([[1,0,1],[1,1,0]])
    print(scale_row(N,0,2))

    L = np.array([[1,0,1],[1,1,0]])
    print(add_rows(L,0,1))

    K = np.array([[1,0,1],[1,0,1],[0,1,1]])
    print(nonzero_in_col(K,1))
    

