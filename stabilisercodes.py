import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit.circuit.library import CZGate
from typing import Literal, Sequence

from binary_RREF import compute_binary_RREF

bit = Literal[0,1]

class Stabiliser:
    """
    Represents a stabiliser (Pauli) operator as binary strings in X and Z.

    A stabiliser is a tensor product of Pauli operators over n qubits.
    Every Pauli matrix is a product of X, Y, Z with a phase 1,-1,i,-i.
    Notes:
        - Y=iXZ so we only need to track the X, Z and phase.
        - X and Z anticommute so in the tensor product we assume each entry is of the form X^aZ^b.
        - X and Z have order 2 so a and b are elements in {0,1}.
    Thus we encode the tensor product as:
        - a phase
        - a binary exponent vector for Z
        - a binary exponent vector for X

    Attributes:
        phase (complex): The global phase of the stabiliser. Must be one of {1, -1, i, -i}.
        z_vec (np.ndarray): Binary exponent vector for Z.
        x_vec (np.ndarray): Binary exponent vector for X.
    """

    def __init__(self,
                 z_vec: list[bit] = [],
                 x_vec: list[bit] = [],
                 phase: complex = 1):
        if phase not in [complex(1), complex(-1), complex(0,1), complex(0,-1)]:
            raise ValueError("Phase must be one of {1, -1, i, -i}.")
        if len(z_vec) == 0 and len(x_vec) == 0:
            raise ValueError("Stabiliser must have some entries.")
        if len(z_vec) == 0:
            z_vec = [0]*len(x_vec)
        if len(x_vec) == 0:
            x_vec = [0]*len(z_vec)
        if len(z_vec) != len(x_vec):
            raise ValueError("Length of z exponent vector and x exponent vector should be the same.")
        
        self.phase = phase
        self.z_vec = z_vec
        self.x_vec = x_vec
        self.num_qubits = len(z_vec)
        self.vec = self.z_vec + self.x_vec

    def multiply_stabilisers(self, stabiliser: 'Stabiliser') -> 'Stabiliser':
        """
        Multiply two stabilisers using exponent vectors.

        Args:
            stabiliser (Stabiliser): Another stabiliser to multiply with self.

        Returns:
            Stabiliser: New stabiliser representing the product.

        Notes:
            - Phase accumulates according to the number of anti-commuting (X,Z) pairs.
        """
        # Phase contribution from anti-commuting X/Z pairs
        p_power = sum(x * z for x, z in zip(stabiliser.x_vec, self.z_vec)) % 2
        new_phase = self.phase * stabiliser.phase * (-1) ** p_power

        # Combine X and Z vectors elementwise using XOR
        # TODO: VSCode does not pick up on the typing without the list[bit] addition
        # Is there a better way to fix the typing?
        new_z : list[bit]= [(z1+z2)%2 for z1, z2 in zip(self.z_vec, stabiliser.z_vec)]
        new_x : list[bit] = [(x1+x2)%2 for x1, x2 in zip(self.x_vec, stabiliser.x_vec)]
        return Stabiliser(new_z, new_x, new_phase)

    def square_stabiliser(self) -> 'Stabiliser':
        """
        Return the square of the stabiliser.

        Returns:
            Stabiliser: The stabiliser multiplied by itself.
        
        Notes:
            - For a valid stabiliser, the square should have phase = 1.
        """
        return self.multiply_stabilisers(self)
    
    def is_valid(self):
        """
        Checks if the stabiliser is valid wrt squaring to id.

        Returns:
            Boolean: True if the stabiliser squares to id. False otherwise.
        """
        return 1 == self.square_stabiliser().phase
    
    def commutes_with(self, stabiliser:Stabiliser):
        """
        Checks if it commutes with a stabiliser.
        Z and X anticommute.
        Only way two ZX stabilisers do not commute is if the phase is changed.
        Thus, just need to check there are an even number of (x,z) pairs.
        """
        d1 = [(z1&x2) for z1, x2 in zip(self.z_vec, stabiliser.x_vec)]
        d2 = [(x1&z2) for x1, z2 in zip(self.x_vec, stabiliser.z_vec)]
        return (sum(d1)+sum(d2))%2 == 0

    def measurement_circuit(self, apply_h_start: bool = True, apply_h_end: bool = True, apply_measure: bool = True) -> QuantumCircuit:
        """
        Create quantum circuit to measure the syndrome of this stabiliser.

        The circuit uses an ancilla qubit to measure the stabiliser operator.

        Args:
            apply_h_start (bool, optional): If True, apply a Hadamard gate to the ancilla at the start.
            apply_h_end (bool, optional): If True, apply a Hadamard gate to the ancilla at the end.
            apply_measure (bool, optional): If True, measure the ancilla qubit into classical bit 0.

        Returns:
            QuantumCircuit: Quantum circuit that measures the stabiliser's syndrome.
        """
        n = self.num_qubits
        qc = QuantumCircuit(n + 1, 1)  # n physical qubits + 1 ancilla
        ancilla = n
        if apply_h_start:
            qc.h(ancilla)
        for idx in range(n):
            if self.z_vec[idx]:
                qc.append(CZGate(), [ancilla,idx])
                # TODO: why is this not rendering correctly?
            if self.x_vec[idx]:
                qc.cx(ancilla,idx)
        if apply_h_end:
            qc.h(ancilla)
        if apply_measure:
            qc.measure(ancilla, 0)
        return qc
    
    def __eq__(self, other : Stabiliser):
        """
        Check if two stabilisers are the same.
        """
        return (self.z_vec == other.z_vec and self.x_vec == other.x_vec and self.phase == other.phase)

    def __repr__(self):
        list = []
        for idx in range(self.num_qubits):
            if self.z_vec[idx] == 1 and self.x_vec[idx] == 1:
                list.append("ZX")
            if self.z_vec[idx] == 1 and self.x_vec[idx] == 0:
                list.append("Z")
            if self.z_vec[idx] == 0 and self.x_vec[idx] == 1:
                list.append("X")
            if self.z_vec[idx] == 0 and self.x_vec[idx] == 0:
                list.append("I")
        if self.phase == 1:
            return f"Stabiliser(Tensor={list})"
        else:
            return f"Stabiliser(Phase={self.phase}, Tensor={list})"

class StabiliserCode:
    """
    Represents a stabiliser code defined by a set of Pauli operators.

    Attributes:
        num_logical_qubits (int): Number of qubits that hold the logical information.
        num_physical_qubits (int): Number of qubits encoding the logical qubits.
        stabilisers (List[Pauli]): List of Pauli operators defining the stabiliser group.
            Each Pauli represents a tensor product over all physical qubits.
            Example: 'XZ' = X tensored with Z.
    """
    def __init__(self,
                 stabilisers: Sequence[Stabiliser]):
        if not stabilisers:
            raise ValueError("Must provide at least one stabiliser.")
        # check the stabilisers are the same length, -id is not in the list and square to id not -id
        stab_len = stabilisers[0].num_qubits
        for stabiliser in stabilisers:
            if stab_len != stabiliser.num_qubits:
                raise ValueError(f"Each stabiliser must have {stab_len} z and x vectors.")
            if stabiliser == Stabiliser(z_vec=[0]*stab_len,x_vec=[0]*stab_len,phase=-1):
                raise ValueError(f"-I is not allowed in list of stabilisers.")
            if not stabiliser.is_valid():
                raise ValueError(f"{stabiliser} does not square to the identity I.")       

        # take stabiliser of space of n physical qubits wrt r stabilisers
        # stabiliser is of dimension 2^{n-r}
        self.num_physical_qubits = stab_len
        self.stabilisers = stabilisers
        self.stab_generators = self.minimal_generating_set()
        self.num_logical_qubits = self.num_physical_qubits - self.stab_generators[0].num_qubits
        self.rate = self.num_logical_qubits / self.num_physical_qubits
        
        non_commuting_pairs = self.non_commuting_pairs()
        self.all_commute = non_commuting_pairs == []
        if not self.all_commute:
            raise ValueError(f"Invalid list of stabilisers. {non_commuting_pairs} do not commute.")

    def non_commuting_pairs(self) -> list[tuple[int,...]]:
        """
        Check the stabilisers in the list commute with one another.
        """
        non_commuting = []
        for idx, s in enumerate(self.stabilisers):
            for t in self.stabilisers[idx+1:]:
                if not s.commutes_with(t):
                    non_commuting.append((s,t))
        return non_commuting
                
    def minimal_generating_set(self) -> list[Stabiliser]:
        """
        Find minimal generating set of stabilisers.
        Note:
            - stabilisers have order 2 and commute
            - the exponent vectors of the stabilisers form a vector space over F2
            - want to check we have a minimal set of stabiliser generators i.e. find a basis
        
        Compute the row reduced echelon form of the code space.
        """
        M = np.array([stabiliser.vec for stabiliser in self.stabilisers],dtype=bool).T
        rref_M = compute_binary_RREF(M).T
        nonzero_rows = [row for row in rref_M if row.any()]
        l = nonzero_rows[0].size
        half = l // 2        
        stab_basis = []
        for b in nonzero_rows:
            bz = b[:int(half)]
            bx = b[int(half):]
            stab_basis.append(Stabiliser(bz,bx))          
        return stab_basis

    def print_stabilisers(self) -> None:
        print("Stabilisers:")
        for idx, stab in enumerate(self.stabilisers):
            print(f"  S{idx}={stab}")
    
    def print_generating_set(self) -> None:
        print("Minimal generating set:")
        for idx, stab in enumerate(self.minimal_generating_set()):
            print(f" B{idx}={stab}")

if __name__ == "__main__":
    print("=== Qiskit Pauli Examples ===")
    pauli1 = Pauli("XZ")
    pauli2 = Pauli("ZZ")
    composed = pauli1.compose(pauli2)
    print(f"{pauli1.to_label()} * {pauli2.to_label()} = {composed.to_label()}")
    
    p = Pauli("XZIX").tensor(Pauli("X"))
    print(f"Tensor product 'XZIX' ⊗ 'X' has {p.num_qubits} qubits")

    print("\n=== Stabiliser Creation and Properties ===")
    stabz1z2 = Stabiliser(z_vec=[0,1], x_vec=[1,0])
    print("Stabiliser z_vec:", stabz1z2.z_vec)
    print("Stabiliser x_vec:", stabz1z2.x_vec)
    print("Is stabiliser valid (squares to identity)?", stabz1z2.is_valid())

    print("\n=== Measurement Circuit ===")
    qc = stabz1z2.measurement_circuit()
    print(qc.draw(output='text'))

    print("\n=== Stabiliser Code Basis Exploration ===")
    stabilisers = [
        Stabiliser([0,0,0],[0,0,0]),
        Stabiliser([0,1,1],[0,0,0]),
        Stabiliser([1,0,1],[0,0,0]),
        Stabiliser([1,1,0],[0,0,0])
    ]
    print(stabilisers[0].commutes_with(stabilisers[2]))
    print(stabilisers[0].vec)
    stabcode = StabiliserCode(stabilisers)
    print("All commute?", stabcode.all_commute)
    basis = stabcode.minimal_generating_set()
    print(stabcode.print_stabilisers())
    print(stabcode.print_generating_set())
    print("Physical qubits:", stabcode.num_physical_qubits)
    print("Logical qubits:", stabcode.num_logical_qubits)
    print("Code rate:", stabcode.rate)

