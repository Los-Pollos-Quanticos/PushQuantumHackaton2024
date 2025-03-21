import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt

from vqd import vqd
from scipy.signal import find_peaks
from collections import defaultdict
from penny_config import HAMILTONIAN
import matplotlib.pyplot as plt

from scipy.linalg import expm

np.random.seed(0)

def analytical_solution(hamiltonian):
    """
    Compute the analytical solution of the Hamiltonian.

    Returns:
        - eig_val: a list of eigenvalues
        - eig_vec: a list of eigenvectors
    """
    Hamiltonian_matrix = qml.matrix(hamiltonian)

    eig_val, eig_vec = np.linalg.eig(Hamiltonian_matrix)

    return eig_val, eig_vec.transpose()

def normalize(amps):
    return amps / np.linalg.norm(amps)

def get_uniform_superposition(eig_vecs):
    """
    Compute the uniform superposition of the eigenvectors.
    Args:
        - eig_vecs: a numpy array representing the eigenvectors

    Returns:
        - vec: a numpy array representing the uniform superposition
    """
    N = eig_vecs.shape[0]
    
    # Ensure there is at least one eigenvector
    if N == 0:
        return np.array([])

    # make sum and multiply by 1/sqrt(N) to normalize
    superposition = np.sum(eig_vecs, axis=0) / np.sqrt(N)
    
    # We apply the normalization again only to round up the values 0.999999.. to 1
    return normalize(superposition)

def get_t_n(T_RMS, N, tol=0.001):
    """
    Generate a list of N positive random numbers from a normal distribution with a given RMS value.
    The function repeats the generation until the RMS of the generated numbers is within a tolerance of tol.

    Args:
        T_RMS: Desired RMS value.
        N: Number of random numbers to generate.
        tol: Tolerance for the RMS difference (default: 1).
        
    Returns:
        t_n: A list of N positive random numbers.
    """
    while True:
        t_n = np.abs(np.random.normal(loc=0, scale=T_RMS, size=N)).tolist()
        calculated_rms = np.sqrt(np.mean(np.square(t_n)))
        if abs(calculated_rms - T_RMS) < tol:
            print(f"Calculated RMS (T_RMS={T_RMS}): {calculated_rms}")
            return t_n

def fun(t,e):
    np.cos((analytical_eig_vals[0]-e)*t/2)**2

def plot_sample(Es,res,title='Rodeo Scan', threshold=0.22, legend=True, num_eig=4,t_n=[]):
    x_values = Es
    y_values = [res[E] for E in Es]

    peaks, _ = find_peaks(y_values,height=threshold)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, marker='o', linestyle='-', label='$P_n$')

    # Add vertical red lines at peaks
    for peak in peaks:
        plt.axvline(x=x_values[peak], color='red', linestyle='--', label=f'Peak at {x_values[peak]:.4f}')

    # Add vertical green line at correct eigenvalues
    for eigenvalue in analytical_eig_vals[:num_eig]:
        plt.axvline(x=eigenvalue, color='green', linestyle='-.', label=f'Eigenvalue at {eigenvalue:.4f}')

    plt.title(title)
    plt.xlabel('E')
    plt.ylabel('$P_N$')
    plt.grid(True)
    if legend:
        plt.legend(loc='best')

    N = len(t_n)
    f_values = []

    for e in x_values:
        f = 1
        for i in range(N):
            f *= np.cos((analytical_eig_vals[0] - e) * t_n[i] / 2) ** 2
        f_values.append(f) 
    f_values = np.array(f_values)

    plt.plot(x_values, f_values, color='orange', alpha=0.5, label='f(E)')
    plt.legend(loc='best')
    plt.show()

def rodeo(N, T_RMS, initial_vec, hamiltonian, start, end, step):
    print(f"Running Rodeo algorithm...")

    M = hamiltonian.num_wires

    t_n = get_t_n(T_RMS, N)
    dev = qml.device("default.qubit", wires=N+M)

    @qml.qnode(dev)
    def circuit(E, t_n):
        for i in range(N):
            qml.Hadamard(wires=i)

        qml.StatePrep(initial_vec, wires=range(N, N + M))
        
        # Suzuki-Trotter
        for i in range(N):
            qml.ControlledQubitUnitary(
                qml.matrix(
                    qml.TrotterProduct(
                        hamiltonian, 
                        -1*t_n[i], 
                        n=2, 
                        order=1
                    )
                ),
                control_wires=[i],
                wires=range(N, N + M)
            )

        for i in range(N):
            qml.PhaseShift(E * t_n[i], wires=[i])

        for i in range(N):
            qml.Hadamard(wires=i)

        return qml.probs(wires=range(N))

    res =  defaultdict(list)
    Engs = np.linspace(start, end, step)
    for E in Engs:
        Pn = circuit(E, t_n)[0]
        res[E] = Pn

    return Engs, res, t_n

if __name__ == "__main__":
    try:
        analytical_eig_vals, analytical_eig_vecs = analytical_solution(HAMILTONIAN)
        analytical_uniform_superposition = get_uniform_superposition(analytical_eig_vecs)
    
        data = np.load('rodeo_data.npz',allow_pickle=True)
        Es = data['Es']
        res = data['res'].item()

        print("Data loaded from rodeo_data.npz")

    except FileNotFoundError:
        N = 2 # number of ancilla qubits
        T_RMS = 2

        # vqd_vec = vqd (
        #     N = HAMILTONIAN.num_wires,
        #     num_layers=5,
        #     num_eig=4,
        #     penalty=2,
        #     operator=HAMILTONIAN,
        #     tolerance=1e-6,
        #     num_iterations=200
        # )

        initial_vec = np.zeros(2**HAMILTONIAN.num_wires)
        initial_vec[26] = 1

        fid = np.abs(np.dot(initial_vec, analytical_eig_vecs[0]))**2
        print(f"Initial fidelity: {fid:.4f}")
        start = -2
        end = -0
        step = 100
        
        t_n = get_t_n(T_RMS, N)
        dev = qml.device("default.qubit", wires=N)

        Es, res, t_n = rodeo(N, T_RMS, initial_vec, HAMILTONIAN, start, end, step)

        # np.savez('rodeo_data.npz', Es=Es, res=res)

    plot_sample(Es, res, title='Rodeo Scan', threshold=0.80, legend=True, num_eig=5,t_n=t_n)