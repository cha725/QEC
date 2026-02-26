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
    def entries(self) -> list[list[int]]:
        return self._entries

    @property
    def array(self) -> NDArray[np.bool_]:
        return self._array
    
    def __array__(self, dtype):
        """
        Allows a binary matrix to be treated like a numpy array.
        """
        return np.array(self._array, dtype=dtype)

    def swap_rows(self, row1: int, row2: int) -> None:
        """
        Swap rows 1 and 2 in the matrix.

        Parameters:
            - row1 (int): The index of the first row to be swapped.
            - row2 (int): The index of the second row to be swapped.
        """
        self._array[[row1, row2], :] = self._array[[row2, row1], :]

    def add_rows(self, source_row: int, target_row: int) -> None:
        """
        Add two rows in the matrix.
        Note: 
            - Add source_row to target_row.
            - The source_row row remains untouched.
        """
        self._array[target_row, :] ^= self._array[source_row, :]

    @cached_property
    def _rref_algorithm(self) -> tuple["BinaryMatrix", list[tuple[int,...]]]:
        """
        Compute the reduced row echelon form (RREF) of a matrix.
        """
        M = BinaryMatrix(self._entries)
        num_rows, num_cols = M.shape
        pivot_coords = []
        start_row = 0
        for col in range(num_cols):
            if start_row >= num_rows:
                break
            # Find pivot rows 
            # i.e. rows below start_row that have a nonzero entry in col_idx
            pivot_rows = [r for r in range(start_row,num_rows) if M.array[r, col]]
            # if there are no rows with a nonzero entry in col, move on
            if not pivot_rows:
                continue
            # take the first pivot row
            pivot_row = pivot_rows.pop(0)
            # if row has nonzero entry in col add pivot row to make zero (working mod2)
            for row in range(num_rows):
                if row != pivot_row and M.array[row,col]:
                    M.add_rows(pivot_row, row)
            # swap pivot row with the next available row
            if pivot_row != start_row:
                M.swap_rows(start_row, pivot_row)
            pivot_coords.append((start_row, col))
            start_row += 1
        return M, pivot_coords
    
    @cached_property
    def rref(self) -> "BinaryMatrix":
        return self._rref_algorithm[0]
    
    @cached_property
    def generator_matrix(self) -> NDArray:
        """
        Return a RREF matrix that has the same rowspan as the binary matrix.
        """
        R = self.rref.array
        return R[~np.all(R == 0, axis=1),:]

    @cached_property
    def _basis(self) -> list[NDArray]:
        """
        Return basis of row space of binary matrix as a list of numpy arrays.
        Only keep those rows that have a nonzero entry.
        """
        return [row.astype(int) for row in self.generator_matrix]
        
    @cached_property
    def basis(self) -> list[list[int]]:
        """
        Return basis of row space of binary matrix as a list of binary lists.
        Only keep those rows that have a nonzero entry.
        """
        return [arr.tolist() for arr in self._basis]
    
    
    def rowspan_matrix(self) -> "BinaryMatrix":
        """
        Return new BinaryMatrix that is reduced and has the same row span.
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
        M, pivots = self._rref_algorithm
        n = M.shape[1]
        pivot_cols = [c for _, c in pivots]
        free_cols = [c for c in range(n) if c not in pivot_cols]
        
        basis = []
        for free in free_cols:
            vec = np.zeros(n, dtype=bool)
            vec[free] = True
            for row, col in reversed(pivots):
                s = np.any(M.array[row, col+1:] & vec[col+1:])
                vec[col] = s
            basis.append(vec)
        
        if basis:
            return BinaryMatrix(basis)
        else:
            return BinaryMatrix([[0]*self.shape[1]])

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