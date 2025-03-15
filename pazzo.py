import pennylane as qml
import numpy as np
from scipy.linalg import kron
import pennylane as qml
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
# # H = (-1.0523 * qml.Identity(0) @ qml.Identity(1) 
# #      + 0.3979 * qml.Identity(0) @ qml.PauliZ(1) 
# #      - 0.3979 * qml.PauliZ(0) @ qml.Identity(1) 
# #      - 0.0112 * qml.PauliZ(0) @ qml.PauliZ(1) 
# #      + 0.1809 * qml.PauliX(0) @ qml.PauliX(1))

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

# Define a deeper and more complex ansatz
def ansatz(params):
    # Initial Hadamard layer to create superposition
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)
    
    # First layer of rotations and entanglement
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])

    # Second layer of rotations and entanglement
    qml.RX(params[2], wires=0)
    qml.RY(params[3], wires=1)
    qml.CNOT(wires=[1, 0])
    qml.CNOT(wires=[0, 1])

    # Add a third layer of rotations and entanglement
    qml.RX(params[4], wires=0)
    qml.RY(params[5], wires=1)
    qml.CNOT(wires=[0, 1])
    
    # Add a fourth layer to further increase expressivity
    qml.RX(params[6], wires=0)
    qml.RY(params[7], wires=1)
    qml.CNOT(wires=[1, 0])
    qml.CNOT(wires=[0, 1])

    # Add additional layers if necessary to match your desired depth and expressivity
    # Keep adding layers of RX, RY, and CNOTs until the model is sufficiently deep
    qml.RX(params[8], wires=0)
    qml.RY(params[9], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 0])

# Use the statevector simulator device
dev = qml.device('default.qubit', wires=2, shots=None)  # Set shots=None for statevector simulation

# Define the cost function (expectation value of the Hamiltonian)
@qml.qnode(dev)
def cost_function(params):
    ansatz(params)
    return qml.expval(H)

# Use the Adam optimizer with a small learning rate for finer convergence
optimizer = AdamOptimizer(stepsize=0.1)
#try other optimizers
# optimizer = qml.GradientDescentOptimizer(stepsize=0.5)

# Initialize random parameters for the ansatz
params = np.random.rand(10)

# Run VQE with a larger number of iterations and monitor convergence
num_iterations = 200
previous_energy = float('inf')
tolerance = 1e-6

for i in range(num_iterations):
    params = optimizer.step(cost_function, params)
    energy = cost_function(params)
    
    # Print the energy every 20 iterations
    if i % 20 == 0:
        print(f"Iteration {i}: Energy = {energy}")
    
    # Early stopping if energy change is below tolerance
    if abs(previous_energy - energy) < tolerance:
        print(f"Converged after {i} iterations.")
        break
    
    previous_energy = energy

# Final energy (ground state energy) and optimized parameters
final_energy = cost_function(params)
print(f"Final ground state energy (VQE): {final_energy}")

# Extract the eigenvector (quantum state) corresponding to the final optimized parameters
@qml.qnode(dev)
def eigenvector_function(params):
    ansatz(params)
    return qml.state()  # Returns the state vector

# Evaluate the quantum state at the final optimized parameters
eigenvector = eigenvector_function(params)

# Print the eigenvector (state vector)
print("Ground state eigenvector (quantum state):")
print(eigenvector)

# Calculate the overlap between the VQE eigenstate and the analytical eigenstate
overlap = np.abs(np.vdot(eigenvector, analytical_eigenstate))**2
print(f"Overlap between VQE eigenstate and analytical eigenstate: {overlap}")

for i in range(3):
    #modify the hamiltonian to penalize the overlap with the eigenvector found
    H = H + 2 * qml.Projector(eigenvector, wires=[0, 1])

    # Use the statevector simulator device
    dev = qml.device('default.qubit', wires=2, shots=None)  # Set shots=None for statevector simulation

    # Define the cost function (expectation value of the Hamiltonian)
    @qml.qnode(dev)
    def cost_function(params):
        ansatz(params)
        return qml.expval(H)

    # Use the Adam optimizer with a small learning rate for finer convergence
    optimizer = AdamOptimizer(stepsize=0.1)
    #try other optimizers
    # optimizer = qml.GradientDescentOptimizer(stepsize=0.5)

    # Initialize random parameters for the ansatz
    params = np.random.rand(10)

    # Run VQE with a larger number of iterations and monitor convergence
    num_iterations = 200
    previous_energy = float('inf')
    tolerance = 1e-6

    for i in range(num_iterations):
        params = optimizer.step(cost_function, params)
        energy = cost_function(params)
        
        # Print the energy every 20 iterations
        if i % 20 == 0:
            print(f"Iteration {i}: Energy = {energy}")
        
        # Early stopping if energy change is below tolerance
        if abs(previous_energy - energy) < tolerance:
            print(f"Converged after {i} iterations.")
            break
        
        previous_energy = energy

    # Final energy (ground state energy) and optimized parameters
    final_energy = cost_function(params)
    print(f"Final ground state energy (VQE): {final_energy}")
    # Extract the eigenvector (quantum state) corresponding to the final optimized parameters
    @qml.qnode(dev)
    def eigenvector_function(params):
        ansatz(params)
        return qml.state()  # Returns the state vector

    # Evaluate the quantum state at the final optimized parameters
    eigenvector = eigenvector_function(params)

    # Print the eigenvector (state vector)
    print("Ground state eigenvector (quantum state):")
    print(eigenvector)

    # calculate the analytical eigenstate form the new H
    H_matrix = H_matrix + 2 * np.outer(ground_state_vector, ground_state_vector)
    eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
    ground_state_energy = eigenvalues[0]
    ground_state_vector = eigenvectors[:, 0]
    analytical_eigenstate = ground_state_vector
    print(f"Analytical ground state energy: {ground_state_energy}")
    print(f"Ground state eigenvector: {ground_state_vector}")
    # Calculate the overlap between the VQE eigenstate and the analytical eigenstate
    overlap = np.abs(np.vdot(eigenvector, analytical_eigenstate))**2
    print(f"Overlap between VQE eigenstate and analytical eigenstate: {overlap}")