from stabilizercodes import Stabiliser, StabiliserCode

class CSSStabiliser(Stabiliser):
    def __init__(self,
                 vec : list[int] = [],
                 z_type : bool = True):
        if z_type:
            z_vec = vec
            x_vec = [0]*len(z_vec)
        elif not z_type:
            x_vec = vec
            z_vec = [0]*len(x_vec)
        super().__init__(z_vec, x_vec, phase=1)

class CSSStabiliserCode(StabiliserCode):
    def __init__(self,
                 z_vecs: list[list[int]] = [],
                 x_vecs: list[list[int]] = []):
        z_stabilisers = [CSSStabiliser(vec, True) for vec in z_vecs]
        x_stabilisers = [CSSStabiliser(vec, False) for vec in x_vecs]
        all_stabilisers = z_stabilisers + x_stabilisers
        super().__init__(all_stabilisers)

