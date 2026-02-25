from stabilizercodes import bit, Stabilizer, StabilizerCode

class CSSStabilizer(Stabilizer):
    """
    Represents a stabilizer in a CSS code.
    A stabilizer in a CSS code has phase=1 and either contains:
        - only I and Z operators; or
        - only I and X operators.
        e.g.    Tensor(X,X,I), Tensor(Z,I,Z,Z) are CSS stabilizers.
                Tensor(X,I,Z) is not a CSS stabilizer.

    Attributes:
        z_type (bool):  True if stabilizer contains only Z and I.
                        False if stabilizer contains only X and I.
        vec (list[int]): Exponent vector of the stabilizer.
    """

    def __init__(self,
                 z_type : bool = True,
                 vec : list[bit] | None = None):
        if vec is None:
            vec = []
        if z_type:
            z_vec = vec
            x_vec : list[bit] = [0]*len(z_vec)
        elif not z_type:
            x_vec = vec
            z_vec : list[bit] = [0]*len(x_vec)
        super().__init__(z_vec, x_vec, phase=1)

class CSSStabilizerCode(StabilizerCode):
    def __init__(self,
                 z_vecs: list[list[bit]] | None = None,
                 x_vecs: list[list[bit]] | None = None):
        if z_vecs is None:
            z_vecs = []
        if x_vecs is None:
            x_vecs = []
        z_stabilizers = [CSSStabilizer(True, vec) for vec in z_vecs]
        x_stabilizers = [CSSStabilizer(False, vec) for vec in x_vecs]
        all_stabilizers = z_stabilizers + x_stabilizers
        super().__init__(all_stabilizers)



