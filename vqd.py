import numpy as np
import pennylane as qml

from pennylane import numpy as np
from penny_config import HAMILTONIAN
from pennylane.optimize import AdamOptimizer

def analytical_solution(hamiltonian):
    """
    Compute the analytical solution of the Hamiltonian.

    Args:
        - hamiltonian: express using the pennylane qml operations

    Returns:
        - eig_val: a list of eigenvalues
        - eig_vec: a list of eigenvectors
    """
    Hamiltonian_matrix = qml.matrix(hamiltonian)

    eig_vals, eig_vecs = np.linalg.eigh(Hamiltonian_matrix)

    # put the eigenvectors in the correct shape, i.e each row is an eigenvector
    eig_vecs = eig_vecs.transpose()

    # sort the eigenvalues and eigenvectors
    # for example, if eig_vals = [3, 1, 2], then eig_vals.argsort() will return [1, 2, 0]
    ordered_idxs = eig_vals.argsort()
    eig_vals = eig_vals[ordered_idxs]
    eig_vecs = eig_vecs[ordered_idxs]

    return eig_vals, eig_vecs


def ansatz(params, num_layers, N):
    for i in range(num_layers):
        for qubit in range(N):
            qml.Rot(params[i, qubit, 0], params[i, qubit, 1], params[i, qubit, 2], wires=qubit)
        
        for qubit in range(N - 1):
            qml.CNOT(wires=[qubit, qubit + 1])

        qml.CNOT(wires=[N - 1, 0])

def vqd(N, num_layers, num_eig, penalty, operator, tolerance=1e-6, num_iterations=300):
    dev = qml.device("default.qubit", wires=N)
    optimizer = AdamOptimizer(stepsize=0.1)

    vqd_eig_vecs = []
    analytical_eig_vals, analytical_eig_vecs = analytical_solution(operator)
    analytical_state = analytical_eig_vecs[0]

    for i in range(num_eig):
        if i > 0 :
            analytical_state = analytical_eig_vecs[i]
            operator = operator + penalty * qml.Projector(eigenvector, wires=range(N))

        print("Analytical eigenvalues: ", analytical_eig_vals[i])
        @qml.qnode(dev)
        def cost_function(params):
            ansatz(params, num_layers, N)
            return qml.expval(operator)

        params = np.random.uniform(low=0, high=2 * np.pi, size=(num_layers, N, 3))

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

        final_energy = cost_function(params)
        print(f"Final ground state energy (VQE): {final_energy}")

        @qml.qnode(dev)
        def eigenvector_function(params):
            ansatz(params, num_layers, N)
            return qml.state()

        eigenvector = eigenvector_function(params)
        vqd_eig_vecs.append(eigenvector)

        overlap = np.abs(np.dot(eigenvector, analytical_state))**2
        print(f"Overlap between VQE eigenstate and analytical eigenstate: {overlap}\n")

    return vqd_eig_vecs

if __name__ == "__main__":
    N = HAMILTONIAN.num_wires
    num_layers = 5
    num_iterations = 300
    tolerance = 1e-6

    penalty = 2

    vqd(
        N=N,
        num_layers=num_layers,
        num_eig=4,
        penalty=penalty,
        operator=HAMILTONIAN,
        tolerance=tolerance,
        num_iterations=num_iterations
    )