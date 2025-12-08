from stabilisercodes import bit
from CSScodes import CSSStabiliserCode

z_vecs = [[0,0,0,1,1,1,1],
          [0,1,1,0,0,1,1],
          [1,0,1,0,1,0,1]]

x_vecs = [[0,0,0,1,1,1,1],
          [0,1,1,0,0,1,1],
          [1,0,1,0,1,0,1]]

# Build the CSS stabilizer code
steane_code = CSSStabiliserCode(z_vecs, x_vecs)

print("=== Steane Code ===")
steane_code.print_stabilisers()
steane_code.print_generating_set()
print("Physical qubits:", steane_code.num_physical_qubits)
print("Logical qubits:", steane_code.num_logical_qubits)
print("Code rate:", steane_code.rate)