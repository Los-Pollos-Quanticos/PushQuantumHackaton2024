import numpy as np

from classiq import *
from classiq.applications.chemistry import (
    ChemistryExecutionParameters,
    HEAParameters,
    Molecule,
    MoleculeProblem,
    UCCParameters,
)
from classiq.execution import (
    ClassiqBackendPreferences,
    ClassiqSimulatorBackendNames,
    ExecutionPreferences,
    OptimizerType,
)

molecule = Molecule(atoms=[("H", (0.0, 0.0, 0)), ("H", (0.0, 0.0, 0.735))])

chemistry_problem = MoleculeProblem(
    molecule=molecule,
    mapping="jordan_wigner",  #'bravyi_kitaev'
    z2_symmetries=True,
    freeze_core=False,
)

operator = chemistry_problem.generate_hamiltonian()
gs_problem = chemistry_problem.update_problem(operator.num_qubits)
print("Your Hamiltonian is", operator.show(), sep="\n")

mat = operator.to_matrix()
w, v = np.linalg.eig(mat)
print("Eigenvalues:", w)
print("Eigenvectors:", v)

serialized_chemistry_model = construct_chemistry_model(
    chemistry_problem=chemistry_problem,
    use_hartree_fock=True,
    ansatz_parameters=UCCParameters(excitations=[1, 2]),
    execution_parameters=ChemistryExecutionParameters(
        optimizer=OptimizerType.COBYLA,
        max_iteration=100,
        initial_point=None,
    ),
)

backend_preferences = ClassiqBackendPreferences(
    backend_name=ClassiqSimulatorBackendNames.SIMULATOR
)

serialized_chemistry_model = set_execution_preferences(
    serialized_chemistry_model,
    execution_preferences=ExecutionPreferences(
        num_shots=1000, backend_preferences=backend_preferences
    ),
)
qprog_ucc = synthesize(serialized_chemistry_model)
result = execute(qprog_ucc).result()
chemistry_result_dict = result[1].value

print("Energy:", chemistry_result_dict["energy"])
eigen = chemistry_result_dict["vqe_result"]["eigenstate"]
print("Eigenstate:", eigen)
