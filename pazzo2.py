import pennylane as qml
from scipy.linalg import kron
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer

# # Define Pauli matrices and identity matrix
# I = np.array([[1, 0], [0, 1]])  # Identity matrix
# X = np.array([[0, 1], [1, 0]])  # Pauli-X matrix
# Y = np.array([[0, -1j], [1j, 0]])  # Pauli-Y matrix
# Z = np.array([[1, 0], [0, -1]])  # Pauli-Z matrix

# # New Hamiltonian components based on the Pauli terms you provided
# H1 = -1.0523 * np.kron(I, I)  
# H2 = 0.3979 * np.kron(I, Z)  
# H3 = -0.3979 * np.kron(Z, I)  
# H4 = -0.0112 * np.kron(Z, Z)  
# H5 = 0.1809 * np.kron(X, X)  

# # Sum all terms to get the total Hamiltonian
# H_matrix = H1 + H2 + H3 + H4 + H5

# Define Pauli matrices and the Identity matrix
I = np.array([[1, 0], [0, 1]])  # Identity matrix
X = np.array([[0, 1], [1, 0]])  # Pauli-X matrix
Y = np.array([[0, -1j], [1j, 0]])  # Pauli-Y matrix
Z = np.array([[1, 0], [0, -1]])  # Pauli-Z matrix

# Define the Hamiltonian terms using Kronecker product (np.kron)
H1 = -12.533 * np.kron(np.kron(np.kron(np.kron(np.kron(I, I), I), I), I), I)
H2 = -1.276 * np.kron(np.kron(np.kron(np.kron(np.kron(Z, I), I), Z), I), I)
H3 = 0.627 * np.kron(np.kron(np.kron(np.kron(np.kron(Z, Z), I), I), I), I)
H4 = -0.875 * np.kron(np.kron(np.kron(np.kron(np.kron(I, Z), I), I), Z), I)
H5 = 0.452 * np.kron(np.kron(np.kron(np.kron(np.kron(I, I), Z), Z), I), I)
H6 = 0.182 * np.kron(np.kron(np.kron(np.kron(np.kron(X, I), X), I), I), I)
H7 = 0.139 * np.kron(np.kron(np.kron(np.kron(np.kron(I, X), I), X), I), I)
H8 = -0.047 * np.kron(np.kron(np.kron(np.kron(np.kron(Y, Y), I), I), I), I)
H9 = 0.209 * np.kron(np.kron(np.kron(np.kron(np.kron(Z, I), Z), I), Z), I)
H10 = -0.154 * np.kron(np.kron(np.kron(np.kron(np.kron(Z, Z), Z), Z), I), I)
H11 = 0.198 * np.kron(np.kron(np.kron(np.kron(np.kron(I, Z), I), Z), Z), Z)
H12 = 0.061 * np.kron(np.kron(np.kron(np.kron(np.kron(X, I), I), X), I), I)
H13 = -0.027 * np.kron(np.kron(np.kron(np.kron(np.kron(I, I), Y), I), Y), I)
H14 = 0.118 * np.kron(np.kron(np.kron(np.kron(np.kron(Z, I), Z), Z), I), Z)

# Sum all Hamiltonian terms
H_matrix = H1 + H2 + H3 + H4 + H5 + H6 + H7 + H8 + H9 + H10 + H11 + H12 + H13 + H14

# H_matrix now represents the full Hamiltonian as a matrix

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)

# Find the ground state (lowest eigenvalue)
ground_state_energy = eigenvalues[0]
ground_state_vector = eigenvectors[:, 0]

print(f"Analytical ground state energy: {ground_state_energy}")
print(f"Ground state eigenvector: {ground_state_vector}")

analytical_eigenstate = ground_state_vector

# # Define the Hamiltonian
# H = (-1.0523 * qml.Identity(0) @ qml.Identity(1) 
#      + 0.3979 * qml.Identity(0) @ qml.PauliZ(1) 
#      - 0.3979 * qml.PauliZ(0) @ qml.Identity(1) 
#      - 0.0112 * qml.PauliZ(0) @ qml.PauliZ(1) 
#      + 0.1809 * qml.PauliX(0) @ qml.PauliX(1))

H = (
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

# Define a deeper and more complex ansatz for six qubits
def ansatz(params):
    # Apply Hadamard gates to all qubits to create superposition
    for i in range(6):
        qml.Hadamard(wires=i)

    # Number of layers (controlled by len(params) / 6)
    layers = len(params) // 6

    for l in range(layers):
        # Apply rotation layer (adding RX and RY gates for each qubit)
        for i in range(6):
            if i % 2 == 0:
                qml.RX(params[l * 6 + i], wires=i)
            else:
                qml.RY(params[l * 6 + i], wires=i)

        # Entanglement layer (CNOTs between neighboring qubits)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[3, 4])
        qml.CNOT(wires=[4, 5])

        # Add a reverse CNOT layer to increase depth and connectivity
        qml.CNOT(wires=[5, 4])
        qml.CNOT(wires=[4, 3])
        qml.CNOT(wires=[3, 2])
        qml.CNOT(wires=[2, 1])
        qml.CNOT(wires=[1, 0])

        # Optionally: Add more advanced gates like controlled-phase gates or Toffoli for depth
        for i in range(5):
            qml.CZ(wires=[i, i + 1])

        # You can add another layer of rotations (using different gates like RZ)
        for i in range(6):
            qml.RZ(params[l * 6 + i], wires=i)


# Use the statevector simulator device for six qubits
dev = qml.device('default.qubit', wires=6, shots=None)

# Define the cost function (expectation value of the Hamiltonian)
@qml.qnode(dev)
def cost_function(params):
    ansatz(params)
    return qml.expval(H)

# Use the Adam optimizer with a learning rate of 0.1
optimizer = AdamOptimizer(stepsize=0.1)

# Initialize random parameters for the ansatz
params = np.random.rand(30)  # Updated for more parameters with six qubits

# Run VQE with a larger number of iterations and monitor convergence
num_iterations = 400
previous_energy = float('inf')
tolerance = 1e-6

for i in range(num_iterations):
    params = optimizer.step(cost_function, params)
    energy = cost_function(params)
    
    if i % 20 == 0:
        print(f"Iteration {i}: Energy = {energy}")
    
    # Early stopping if energy change is below tolerance
    if abs(previous_energy - energy) < tolerance:
        print(f"Converged after {i} iterations.")
        break
    
    previous_energy = energy

# Final energy and optimized parameters
final_energy = cost_function(params)
print(f"Final ground state energy (VQE): {final_energy}")

# Extract the eigenvector (quantum state) corresponding to the final optimized parameters
@qml.qnode(dev)
def eigenvector_function(params):
    ansatz(params)
    return qml.state()

# Evaluate the quantum state at the final optimized parameters
eigenvector = eigenvector_function(params)
print("Ground state eigenvector (quantum state):")
# print(eigenvector)

# Iterative penalization of eigenvector overlap and Hamiltonian update
for iteration in range(3):
    H = H + 10 * qml.Projector(eigenvector, wires=[0, 1, 2, 3, 4, 5])
    print(f"Updated Hamiltonian for iteration {iteration+1}.")

    # Re-optimize with the updated Hamiltonian
    params = np.random.rand(30)  # Reset parameters for each iteration
    previous_energy = float('inf')

    for i in range(num_iterations):
        params = optimizer.step(cost_function, params)
        energy = cost_function(params)
        
        if i % 20 == 0:
            print(f"Iteration {i}: Energy = {energy}")
        
        if abs(previous_energy - energy) < tolerance:
            print(f"Converged after {i} iterations.")
            break

        previous_energy = energy

    # Final energy and eigenvector
    final_energy = cost_function(params)
    eigenvector = eigenvector_function(params)

    print(f"Final ground state energy (VQE) after penalization: {final_energy}")
    print("Ground state eigenvector (quantum state):")
    # print(eigenvector)
