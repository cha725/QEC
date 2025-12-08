from stabilizercodes import Stabiliser, StabiliserCode
from typing import Literal
from stabilizercodes import bit

class CSSStabiliser(Stabiliser):
    """
    Represents a stabiliser in a CSS code.
    A stabiliser in a CSS code has phase=1 and either contains:
        - only I and Z operators; or
        - only I and X operators.
        e.g.    Tensor(X,X,I), Tensor(Z,I,Z,Z) are CSS stabilisers.
                Tensor(X,I,Z) is not a CSS stabiliser.

    Attributes:
        z_type (bool):  True if stabiliser contains only Z and I.
                        False if stabiliser contains only X and I.
        vec (list[int]): Exponent vector of the stabiliser.
    """

    def __init__(self,
                 z_type : bool = True,
                 vec : list[bit] = []):
        if z_type:
            z_vec = vec
            x_vec : list[bit] = [0]*len(z_vec)
        elif not z_type:
            x_vec = vec
            z_vec : list[bit] = [0]*len(x_vec)
        super().__init__(z_vec, x_vec, phase=1)

class CSSStabiliserCode(StabiliserCode):
    def __init__(self,
                 z_vecs: list[list[bit]] = [],
                 x_vecs: list[list[bit]] = []):
        z_stabilisers = [CSSStabiliser(True, vec) for vec in z_vecs]
        x_stabilisers = [CSSStabiliser(True, vec) for vec in x_vecs]
        all_stabilisers = z_stabilisers + x_stabilisers
        super().__init__(all_stabilisers)

