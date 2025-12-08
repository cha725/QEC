import numpy as np
from numpy.typing import NDArray

def row_swap(M: NDArray, i: int, j: int) -> NDArray:
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

def scale_row(M: NDArray, i: int, c: int) -> NDArray:
    """
    Scale a row in a matrix by a constant integer.
    Note: will be used only for binary matrices over F_2.

    Parameters:
        - M (NDArray): The matrix on which to perform the scalar multiplication.
        - i (int): The index of the row to scale.
        - c (int): The constant to multiple the row with.
    """
    M = M.copy()
    num_rows = M.shape[0]
    C = np.identity(num_rows)
    C[i][i] = c
    return np.matmul(C,M)

def add_rows(M: NDArray, i: int, j: int) -> NDArray:
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
    M[[i],:] += M[[j],:]
    return M

def nonzero_in_col(M: NDArray, i: int) -> list[bool]:
    """
    Return a list of row indices of a matrix that are nonzero in a specific column.

    Parameters:
        - M (NDArray): The matrix to check.
        - i (int): The column index to check.
    """
    M = M.copy()
    col = M[:,[i]].T
    return [x != 0 for x in col]




if __name__ == "__main__":
    
    M = np.array([[1,2,3],[4,5,6]])
    print(row_swap(M,0,1))
    N = np.array([[1,2,3],[4,5,6]])
    print(scale_row(N,0,2))
    L = np.array([[1,2,3],[4,5,6]])
    print(add_rows(L,0,1))

    K = np.array([[1,0,3],[4,0,6],[0,1,2]])
    print(nonzero_in_col(K,1))
    

