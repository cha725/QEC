# Error Correction Notes

This repository is a work in progress collecting theoretical explanations and Python implementations of classical and quantum error correction. 
It is intended to be suitable for undergraduates with basic linear algebra and probability knowledge.

## Contents

The project is currently organised into three Jupyter notebooks with accompanying Python files.

### Python files

1. binary__RREF.py: contains BinaryMatrix class to compute row reduced echelon form of a binary matrix.
2. linearcodes.py: contains LinearCode class and RepetitionCode, HammingCode, LDPCCode, RandomLDPC code subclasses.
3. belief_propagation.py: contains MessagePassing and BeliefPropagation class.
4. stabilizercodes.py: contains Stabilizer and StabilizerCode class.

### Jupyter Notebooks

1. **Classical Error Correction**

This notebook introduces the motivation for classical error correction and the structure of linear codes. 
It covers the definition of linear codes, Hamming codes and LDPC codes. 
It includes small implementations of these codes that rely on the functions in linearcodes.py.
The file linearcodes.py provides a LinearCodes class that constructs generator and parity check matrices and also builds the associated Tanner graph.

2. **Classical Decoding**

This notebook introduces classical decoding methods. 
It explains maximum likelihood decoding with a Python implementation and an outline of the underlying probability theory. 
It also gives an introduction to message passing and belief propagation on Tanner graphs.
The complementary file belief_propagation.py contains a class implementing message passing and belief propagation for linear codes.

3. **Quantum Error Correction**

This notebook is intended to introduce surface codes from scratch. 
It currently develops stabilizer groups and Pauli matrices. 
The accompanying files CSScodes.py and stabilizercodes.py define classes for CSS codes and stabilizer codes.
