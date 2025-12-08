import numpy as np
from numpy.typing import NDArray

def swap_rows(M: NDArray[np.bool_], i: int, j: int) -> NDArray[np.bool_]:
    """
    Swap two rows in a matrix.

    Parameters:
        - m (NDArray): The matrix to switch the rows in.
        - i (int): The index of the first row to be swapped.
        - j (int): The index of the second row to be swapped.
    """
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

def compute_binary_RREF(M: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """
    Compute the reduced row echelon form (RREF) of a matrix.

    Parameters:
        - M (NDArray): The matrix to change to RREF.
    """
    M = M.copy()
    n_rows, n_cols = M.shape
    next_row_idx = 0
    for col_idx in range(n_cols):
        candidate_rows = nonzero_in_col(M, col_idx)
        candidate_pivot_rows = [row for row in candidate_rows if row >= col_idx]
        if not candidate_pivot_rows:
            continue
        pivot_row = candidate_pivot_rows.pop(0)
        for row in candidate_rows:
            if row != pivot_row:
                add_rows(M, row, pivot_row)
        if pivot_row != col_idx:
            swap_rows(M, next_row_idx, pivot_row)
        next_row_idx += 1
        if next_row_idx >= n_rows:
            break
    return M

if __name__ == "__main__":
    
    M = np.array([[1,0,1],[1,1,0]])
    print(swap_rows(M,0,1))

    N = np.array([[1,0,1],[1,1,0]])
    print(add_rows(N,0,1))

    L = np.array([[1,0,1,0],[1,0,1,1],[0,1,1,0],[1,0,1,0],[0,0,1,1]])
    print(nonzero_in_col(L,1))
    
    print(compute_binary_RREF(L))

