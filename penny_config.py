import pennylane as qml

# HAMILTONIAN = (-1.0523 * qml.Identity(0) @ qml.Identity(1) 
#      + 0.3979 * qml.Identity(0) @ qml.PauliZ(1)
#      - 0.3979 * qml.PauliZ(0) @ qml.Identity(1)
#      - 0.0112 * qml.PauliZ(0) @ qml.PauliZ(1)
#      + 0.1809 * qml.PauliX(0) @ qml.PauliX(1))

HAMILTONIAN = (
    -12.533 * qml.Identity(0) @ qml.Identity(1) @ qml.Identity(2) @ qml.Identity(3) @ qml.Identity(4) @ qml.Identity(5)
    - 1.276 * qml.PauliZ(0) @ qml.Identity(1) @ qml.Identity(2) @ qml.PauliZ(3) @ qml.Identity(4) @ qml.Identity(5)
    + 0.627 * qml.PauliZ(0) @ qml.PauliZ(1) @ qml.Identity(2) @ qml.Identity(3) @ qml.Identity(4) @ qml.Identity(5)
    - 0.875 * qml.Identity(0) @ qml.PauliZ(1) @ qml.Identity(2) @ qml.Identity(3) @ qml.PauliZ(4) @ qml.Identity(5)
    + 0.452 * qml.Identity(0) @ qml.Identity(1) @ qml.PauliZ(2) @ qml.PauliZ(3) @ qml.Identity(4) @ qml.Identity(5)
    + 0.182 * qml.PauliX(0) @ qml.Identity(1) @ qml.PauliX(2) @ qml.Identity(3) @ qml.Identity(4) @ qml.Identity(5)
    + 0.139 * qml.Identity(0) @ qml.PauliX(1) @ qml.Identity(2) @ qml.PauliX(3) @ qml.Identity(4) @ qml.Identity(5)
    - 0.047 * qml.PauliY(0) @ qml.PauliY(1) @ qml.Identity(2) @ qml.Identity(3) @ qml.Identity(4) @ qml.Identity(5)
    + 0.209 * qml.PauliZ(0) @ qml.Identity(1) @ qml.PauliZ(2) @ qml.Identity(3) @ qml.PauliZ(4) @ qml.Identity(5)
    - 0.154 * qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3) @ qml.Identity(4) @ qml.Identity(5)
    + 0.198 * qml.Identity(0) @ qml.PauliZ(1) @ qml.Identity(2) @ qml.PauliZ(3) @ qml.PauliZ(4) @ qml.PauliZ(5)
    + 0.061 * qml.PauliX(0) @ qml.Identity(1) @ qml.Identity(2) @ qml.Identity(3) @ qml.PauliX(4) @ qml.Identity(5)
    - 0.027 * qml.Identity(0) @ qml.Identity(1) @ qml.PauliY(2) @ qml.Identity(3) @ qml.PauliY(4) @ qml.Identity(5)
    + 0.118 * qml.PauliZ(0) @ qml.Identity(1) @ qml.PauliZ(2) @ qml.PauliZ(3) @ qml.Identity(4) @ qml.PauliZ(5)
) 

hamiltonian_matrix = qml.matrix(HAMILTONIAN)