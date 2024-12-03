# Rodeo Algorithm for Estimating Molecular Spectra - PushQuantum Hackathon 2024

<div style="display: flex; justify-content: center; align-items: center; gap: 20px;">
    <img src="https://docs.classiq.io/resources/pushquantum_logo.png" alt="PushQuantum Logo" style="width: 200px; height: auto;" />
    <img src="https://avatars.githubusercontent.com/u/190467285?s=400&u=f1cd2ff965acf448e997e31e12778ec64f23b372&v=4" alt="Team Los Pollos Quanticos" style="width: 200px; height: auto;" />
    <img src="https://docs.classiq.io/resources/classiq-logo.svg" alt="Classiq Logo" style="width: 200px; height: auto;" />
</div>

## Team: Los Pollos Quanticos
Welcome to the repository for our implementation of the Rodeo Algorithm for Estimating Molecular Spectra, developed during the Classiq Challenge at the PushQuantum Hackathon 2024.

## Challenge Overview
### Objective
The challenge involved implementing the Rodeo Algorithm, a quantum algorithm designed to estimate the eigenvalues of molecular Hamiltonians. This was a two-part task:

- Implementing the algorithm for the simpler Hamiltonian of the H₂ molecule.

- Extending the implementation to handle a more complex Hamiltonian representing the H₂O molecule.
The Rodeo Algorithm is a promising quantum approach for refining eigenvalue estimates, leveraging quantum phase estimation and iterative techniques to isolate true energy levels.

For more details, refer to the [Rodeo Algorithm Paper](https://arxiv.org/pdf/2009.04092).

## Solution Highlights
### Warm-Up
We began by preparing auxiliary functions for:

- State preparation (prep) to initialize qubits in the |-> state.

- Time evolution using Suzuki-Trotter decomposition (trotter).
- Controlled application of Hamiltonian evolution (apply_control).

### Part 1: H₂ Molecule
- Hamiltonian: We used the given Pauli representation of the H₂ molecular Hamiltonian.
- Analytical Solution: Computed eigenvalues and eigenvectors using numpy.
- Quantum Implementation:
    - Prepared the uniform superposition of eigenstates.
    - Iteratively refined energy estimates using the Rodeo Algorithm.
    - Plotted Averaged Energy Population Distribution to visualize peaks corresponding to eigenvalues.

### Part 2: H₂O Molecule
- Hamiltonian: A more complex 6-qubit representation of the H₂O molecule.
- Adaptations:
  - Extended state preparation for the 6-qubit system.
  - Increased the number of control qubits and iterations.
  - Identified the five lowest energy levels of the Hamiltonian.