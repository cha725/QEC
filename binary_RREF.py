import numpy as np
from scipy.linalg import null_space

from functools import cached_property
from scipy.linalg import null_space

from numpy.typing import NDArray

class BinaryMatrix:

    def __init__(self,
                 matrix: NDArray[np.bool_]):
        self.matrix = matrix.copy()
        self.num_rows = self.matrix.shape[0]
        self.num_cols = self.matrix.shape[1]


    def swap_rows(self , i: int, j: int) -> None:
        """
        Swap rows i and j in the matrix.

        Parameters:
            - i (int): The index of the first row to be swapped.
            - j (int): The index of the second row to be swapped.
        """
        self.matrix[[i,j], :] = self.matrix[[j,i], :]

    def add_rows(self , source_idx: int, target_idx: int) -> None:
        """
        Add two rows in the matrix.
        Note: 
            - Add source_idx row to row target_idx_row.
            - The source_idx row remains untouched.
        """
        self.matrix[[target_idx],:] ^= self.matrix[[source_idx],:]

    def nonzero_in_col(self , idx: int) -> list[int]:
        """
        Return a list of row indices of a matrix that are nonzero in a specific column.

        Parameters:
            - idx (int): The column index to check.
        """
        return [int(idx) for idx in list(np.flatnonzero(self.matrix[:,idx]))]

    @cached_property
    def rref(self ) -> NDArray[np.bool_]:
        """
        Compute the reduced row echelon form (RREF) of a matrix.
        """
        M = BinaryMatrix(self.matrix)
        num_cols = M.num_cols
        next_row_idx = 0
        for col_idx in range(num_cols):
            candidate_rows = M.nonzero_in_col(col_idx)
            candidate_pivot_rows = [row for row in candidate_rows if row >= col_idx]
            if not candidate_pivot_rows:
                continue
            pivot_row = candidate_pivot_rows.pop(0)
            for row in candidate_rows:
                if row != pivot_row:
                    M.add_rows(pivot_row, row)
            if pivot_row != col_idx:
                M.swap_rows(next_row_idx, pivot_row)
            next_row_idx += 1
            if next_row_idx >= M.num_rows:
                break
        return M.matrix    
    
    @property
    def rank(self) -> int:
        """
        Return rank of matrix over F2.
        """
        rank = 0
        matrix = self.rref
        for row_idx in range(self.num_rows):
            if matrix[row_idx,:].any():
                rank += 1
        return rank
    
    @cached_property
    def generator_matrix(self) -> NDArray[np.bool_]:
        return self.rref[~np.all(self.rref == 0, axis=1),:]
    
    def non_identity_part_gen(self):
        return self.generator_matrix[:, range(self.rank, self.num_cols)]

    @cached_property
    def kernel(self) -> NDArray:
        M = self.rref
        m = M.shape[0]
        n = M.shape[1]

        pivot_positions = []
        pivot_cols = set()
        for row_idx in range(m):
            row = M[row_idx,:]
            for col_idx in range(n):
                if row[col_idx] == 1:
                    pivot_positions.append((row_idx, col_idx))
                    pivot_cols.add(col_idx)
                    break

        free_cols = [col_idx for col_idx in range(n) if col_idx not in pivot_cols]
        if not free_cols:
            return np.array([0 for _ in range(n)])
        pivot_positions.sort(key=lambda rc: rc[1])
        basis = []
        for fc_idx in free_cols:
            v = [0 for _ in range(n)]
            v[fc_idx] = 1
            for r, c in reversed(pivot_positions):
                s = 0
                row = M[r]
                for j in range(c + 1, n):
                    if row[j] and v[j]:
                        s ^= 1
                v[c] = s
            basis.append(v)

        return np.array(basis)
    
    @cached_property
    def parity_check_matrix(self) -> NDArray:
        return self.kernel


if __name__ == "__main__":
    
    M = BinaryMatrix(np.array([[1,0,1,0],[1,0,1,1],[0,1,1,0],[1,0,1,0],[0,0,1,1]]))

    print(f"Matrix: {M.matrix}")
    M.swap_rows(0,1)
    print(f"Swap rows 0 and 1: {M.matrix}")
    M.add_rows(1,0)
    print(f"Add row 1 to row 0: {M.matrix}")
    print(f"Nonzero entries in column 2: {M.nonzero_in_col(2)}")
    print(f"RREF form: {M.rref}")
    print(f"Generator matrix: {M.generator_matrix}")
    print(f"Non-identity part: {M.non_identity_part_gen()}")
    print(f"Rank of M = {M.rank}")
    print(f"Kernel of M: {M.kernel}")
    print(f"Parity check matrix: {M.parity_check_matrix}")


    M = BinaryMatrix(np.array([[1,0,1,0],[1,0,1,0],[0,1,1,0],[1,0,1,0],[0,0,1,0]]))

    print(f"Matrix: {M.matrix}")
    M.swap_rows(0,1)
    print(f"Swap rows 0 and 1: {M.matrix}")
    M.add_rows(1,0)
    print(f"Add row 1 to row 0: {M.matrix}")
    print(f"Nonzero entries in column 2: {M.nonzero_in_col(2)}")
    print(f"RREF form: {M.rref}")
    print(f"Generator matrix: {M.generator_matrix}")
    print(f"Non-identity part: {M.non_identity_part_gen()}")
    print(f"Rank of M = {M.rank}")
    print(f"Kernel of M: {M.kernel}")
    print(f"Parity check matrix: {M.parity_check_matrix}")
    

