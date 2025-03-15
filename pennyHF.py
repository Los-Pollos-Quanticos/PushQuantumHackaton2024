import pennylane as qml
from pennylane import numpy as np

# Define the Hamiltonian
H = (-1.0523 * qml.Identity(0) @ qml.Identity(1) 
     + 0.3979 * qml.Identity(0) @ qml.PauliZ(1) 
     - 0.3979 * qml.PauliZ(0) @ qml.Identity(1) 
     - 0.0112 * qml.PauliZ(0) @ qml.PauliZ(1) 
     + 0.1809 * qml.PauliX(0) @ qml.PauliX(1))

# Number of qubits in the system
n_qubits = 2

# Initialize the quantum device
dev = qml.device("default.qubit", wires=n_qubits)

# Hartree-Fock state for a 2-qubit system (both electrons in the lowest orbitals)
# In this case, we can assume the HF state as |11>, meaning both qubits are occupied.
hf_state = np.array([1, 1])

@qml.qnode(dev)
def hf_circuit():
    # Prepare the HF state: |11> = [1, 1]
    qml.BasisState(hf_state, wires=[0, 1])
    
    # Measure the expectation value of the Hamiltonian
    return qml.expval(H)

# Run the Hartree-Fock circuit
hf_energy = hf_circuit()
print(f"HF state energy: {hf_energy}")
