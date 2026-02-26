import numpy as np
from functools import cached_property
from itertools import product
from numpy.typing import NDArray

class BinaryMatrix:
    """
    Represents a matrix over F2.

    Parameters:
        - rows (list[list[int]]): entry [i,j] of the matrix is rows[i,j]
            Any entry in rows that is not 0 will be treated as a 1.
    """
    def __init__(self,
                 rows: list[list[int]]):
        num_cols = len(rows[0])
        if any(len(row) != num_cols for row in rows):
            raise ValueError(f"All rows must have the same length.")
        self._rows = rows
        self.bool_matrix = np.array(self, dtype=bool, copy=True)
        self.shape = self.bool_matrix.shape
        self.num_rows = self.shape[0]
        self.num_cols = self.shape[1]

    @property
    def array(self) -> NDArray[np.int_]:
        return np.array(self, dtype=int)
    
    def __array__(self, dtype, *, copy=True):
        """
        Allows a binary matrix to be treated like a numpy array.
        """
        arr = np.array(self._rows, dtype=dtype)
        if copy:
            return arr.copy()
        return arr

    @cached_property
    def _compute_rref(self) -> tuple[NDArray[np.bool_], list[tuple[int,int]]]:
        """
        Compute the reduced row echelon form (RREF) of a matrix.

        Returns:
            - NDArray[np.bool_]: RREF matrix as a boolean numpy array
            - list[tuple[int,int]]: list of pivot coordinates used in the RREF algorithm
        """
        M = self.bool_matrix.copy()
        num_rows, num_cols = M.shape
        pivot_coords: list[tuple[int,int]] = []
        start_row = 0
        for col in range(num_cols):
            if start_row >= num_rows:
                break
            # Find pivot rows 
            # i.e. rows below start_row that have a nonzero entry in col_idx
            pivot_rows = [r for r in range(start_row,num_rows) if M[r, col]]
            # if there are no rows with a nonzero entry in col, move on
            if not pivot_rows:
                continue
            # take the first pivot row
            pivot_row = pivot_rows.pop(0)
            # if row has nonzero entry in col add pivot row to make zero (working mod2)
            for row in range(num_rows):
                if row != pivot_row and M[row,col]:
                    M[[row], :] ^= M[[pivot_row], :]
            # swap pivot row with the next available row
            if pivot_row != start_row:
                M[[start_row, pivot_row], :] = M[[pivot_row, start_row], :]
            pivot_coords.append((start_row, col))
            start_row += 1
        return M, pivot_coords
    
    @cached_property
    def rref(self) -> "BinaryMatrix":
        """
        Returns:
            - BinaryMatrix: the reduced row echelon form (RREF) of the matrix.
        """
        int_rref = np.array(self._compute_rref[0], dtype=int)
        rows = int_rref.tolist()
        return BinaryMatrix(rows)
    
    @cached_property
    def basis(self) -> list[list[int]]:
        """
        Returns:
            - list[list[int]]: a basis of the row space of the matrix.
        """
        rows = self.rref.array
        return [row.tolist() for row in rows if np.any(row)]
        
    @cached_property
    def rowspace(self) -> "BinaryMatrix":
        """
        Returns:
            - BinaryMatrix: a RREF matrix that has the same rowspan as the matrix.
        """
        return BinaryMatrix(self.basis)

    @property
    def rank(self) -> int:
        """
        Return rank of matrix over F2.
        """
        return len(self._basis)

    def num_rowspan_elements(self) -> int:
        """
        Return the number of elements in the span of the basis.
        """ 
        return 2**self.rank
    
    def _rowspan_elements(self) -> list[NDArray]:
        """
        Return a list of elements in the span of the basis.
        """
        basis = self._basis
        if not basis:
            return [np.zeros(self.shape[1], dtype=int)]
        elements = []
        for coeffs in product([0,1], repeat=self.rank):
            elt = np.zeros(self.shape[1], dtype=int)
            for coeff, vec in zip(coeffs, basis):
                if coeff:
                    elt ^= vec
            elements.append(elt)
        return elements

    def rowspan_elements(self):
        """
        Return elements in span of the basis as a list of binary lists.
        """               
        return [arr.tolist() for arr in self._rowspan_elements()]        

    @cached_property
    def nullspace(self) -> "BinaryMatrix":
        """
        Returns:
            - BinaryMatrix: a basis for the nullspace of the matrix over F2.

        Method:
        1. Compute the RREF of M. This gives:
            - pivot columns which correspond to dependent variables
            - non pivot columns which correspond to free variables
        2. For each free variable:
            - set that free variable = 1
            - set all other free variables = 0
            - compute the pivot variables by back substitution in RREF
              (each pivot variable is the xor of the free variables to its right)
        3. Collect each resulting vector. These form a basis of the nullspace.
        """
        rref, pivots = self._compute_rref
        pivot_cols = [col for _, col in pivots]
        free_cols = [col for col in range(self.num_cols) if col not in pivot_cols]
        basis_vectors = []
        for free in free_cols:
            vec = np.zeros(self.num_cols, dtype=bool)
            vec[free] = True
            for row, pivot_col in reversed(pivots):
                row_after_pivot_col = rref[row, pivot_col+1:]
                vec_after_pivot_col = vec[pivot_col+1:]
                vec[pivot_col] = np.any(row_after_pivot_col & vec_after_pivot_col)
            basis_vectors.append(vec)
        if basis_vectors:
            return BinaryMatrix(basis_vectors)
        else:
            return BinaryMatrix([[0]*self.num_cols])

    def __repr__(self):
        return f"Binary matrix: \n {self.array}."



if __name__ == "__main__":
    
    M = BinaryMatrix([[1,0,1,0],[1,0,1,1],[0,1,1,0],[1,0,1,0],[0,0,1,1]])

    print(f"\n=== Matrix: {M.array}")
    M.swap_rows(0,1)
    print(f"Swap rows 0 and 1: {M.array}")
    M.add_rows(1,0)
    print(f"Add row 1 to row 0: {M.array}")
    print(f"RREF: {M.rref.array}")
    print(f"Generator matrix: {M.generator_matrix}")
    print(f"Rank of M = {M.rank}")
    print(f"Nullspace of M: {M.nullspace.array}")


    M = BinaryMatrix([[1,0,1,0],[1,0,1,0],[0,1,1,0],[1,0,1,0],[0,0,1,0]])

    print(f"\n=== Matrix: {M.array}")
    M.swap_rows(0,1)
    print(f"Swap rows 0 and 1: {M.array}")
    M.add_rows(1,0)
    print(f"Add row 1 to row 0: {M.array}")
    print(f"RREF: {M.rref.array}")
    print(f"Generator matrix: {M.generator_matrix}")
    print(f"Rank of M = {M.rank}")
    print(f"Nullspace of M: {M.nullspace.array}")
    
    M = BinaryMatrix(
        [[0, 0, 0, 0, 0], 
         [0, 0, 0, 0, 1], 
         [0, 0, 0, 1, 0], 
         [0, 0, 0, 1, 1], 
         [1, 0, 1, 0, 0], 
         [1, 0, 1, 0, 1], 
         [1, 0, 1, 1, 0], 
         [1, 0, 1, 1, 1]]
         )
    
    print(f"\n=== Matrix: {M.array}")
    M.swap_rows(0,1)
    print(f"Swap rows 0 and 1: {M.array}")
    M.add_rows(1,0)
    print(f"Add row 1 to row 0: {M.array}")
    print(f"RREF: {M.rref.array}")
    print(f"Generator matrix: {M.generator_matrix}")
    print(f"Rank of M = {M.rank}")
    print(f"Nullspace of M: {M.nullspace.array}")