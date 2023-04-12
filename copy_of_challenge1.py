# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
# Import standard Qiskit libraries
from qiskit import QuantumCircuit, transpile, Aer, IBMQ
from qiskit.providers.aer import StatevectorSimulator
from qiskit.primitives import Estimator
from qiskit.utils import QuantumInstance
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from qiskit.providers.ibmq import least_busy
from qiskit_ibm_runtime import Options
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator, Session

# Import Qiskit libraries for VQE
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver, VQE
from qiskit.algorithms.optimizers import COBYLA, SLSQP, L_BFGS_B, SPSA
# Import Qiskit Nature libraries
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.drivers import UnitsType, Molecule  
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem
from qiskit_nature.second_q.mappers import QubitConverter, JordanWignerMapper, ParityMapper, BravyiKitaevMapper
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.algorithms import VQEUCCFactory, GroundStateEigensolver
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD, PUCCD
from qiskit import *

# Setup the molecule
atom= 'Mn, [0., 0., 0.]; C, [0., 0., 1.0]; O, [1.184433,0.,0.999161]; O, [-1.184433,0.,0.999161]'                              
# Initiate the PySCF driver
driver = PySCFDriver(atom = atom, charge=0,spin=1, unit=UnitsType.ANGSTROM, basis='sto-3g')
molecule=driver.run()

print('Number of molecular orbitals: {}'.format(molecule.num_spatial_orbitals))
print('Number of alpha electrons: {}'.format(molecule.num_alpha))
print('Number of beta electrons: {}'.format(molecule.num_beta))
print('Number of particles: {}'.format(molecule.num_particles))

# Perform the setup calculation using PySCF
pes_problem = ElectronicStructureProblem(driver)
# Get the second quantized Hamiltonian
second_q_op = pes_problem.second_q_ops()

# Commented out IPython magic to ensure Python compatibility.
# %%capture cap --no-stderr
# print(second_q_op[0])

with open('output.txt', 'w') as f:
    f.write(cap.stdout)

# Use Jordan-Wigner mapping to transform the Hamiltonian into qubitized form
qubit_converter = QubitConverter(JordanWignerMapper())
qubit_op = qubit_converter.convert(second_q_op[0])

# Commented out IPython magic to ensure Python compatibility.
# %%capture cap --no-stderr
# print(qubit_op)

with open('output1.txt', 'w') as f:
    f.write(cap.stdout)

# Use Parity  mapping to transform the Hamiltonian into qubitized form
qubit_converter1 = QubitConverter(ParityMapper(), two_qubit_reduction=True)
qubit_op1 = qubit_converter1.convert(second_q_op[0], num_particles=molecule.num_particles)

# Commented out IPython magic to ensure Python compatibility.
# %%capture cap --no-stderr
# print(qubit_op1)

with open('output2.txt', 'w') as f:
    f.write(cap.stdout)

# Use BravyiKitaevMappev mapping to transform the Hamiltonian into qubitized form
qubit_converter2 = QubitConverter(BravyiKitaevMapper(), two_qubit_reduction=True)
qubit_op2 = qubit_converter2.convert(second_q_op[0],num_particles=molecule.num_particles)

# Commented out IPython magic to ensure Python compatibility.
# %%capture cap --no-stderr
# print(qubit_op2)

with open('output3.txt', 'w') as f:
    f.write(cap.stdout)

# Use classical approach to solve the problem
sol = NumPyMinimumEigensolver().compute_minimum_eigenvalue(qubit_op1)
real_solution = molecule.interpret(sol)
real_solution.groundenergy

# Set up the calculation for VQE
# Set the optimizer  
optimizer = SLSQP(maxiter=1000) 

# Set number of spin-orbitals
num_spin_orbitals=64

# Set the number of particles
num_particles=(24,23)

# Set the initial state as the Hartree-Fock state in Qiskit convention
initial_state = HartreeFock(num_spin_orbitals, num_particles, qubit_converter1)

# Set up the calculation for VQE
# Set the optimizer  
optimizer = SLSQP(maxiter=150) 

# Set number of spin-orbitals
num_spin_orbitals1=64

# Set the number of particles
num_particles=(24,23)

# Set the initial state as the Hartree-Fock state in Qiskit convention
initial_state1 = HartreeFock(num_spin_orbitals1, num_particles, qubit_converter)

# Commented out IPython magic to ensure Python compatibility.
# %%capture cap --no-stderr
# print(initial_state)

with open('output4.txt', 'w') as f:
    f.write(cap.stdout)

# Commented out IPython magic to ensure Python compatibility.
# %%capture cap --no-stderr
# print(initial_state1)

with open('output41.txt', 'w') as f:
    f.write(cap.stdout)

# Set the UCCSD ansatz
ansatz = UCCSD(num_spin_orbitals,num_particles,qubit_converter1,initial_state=initial_state)

# Set the UCCSD ansatz
ansatz1 = UCCSD(num_spin_orbitals1,num_particles,qubit_converter,initial_state=initial_state1)

# Commented out IPython magic to ensure Python compatibility.
# %%capture cap --no-stderr
# print(ansatz)

with open('output5.txt', 'w') as f:
    f.write(cap.stdout)

# Commented out IPython magic to ensure Python compatibility.
# %%capture cap --no-stderr
# print(ansatz1)

with open('output51.txt', 'w') as f:
    f.write(cap.stdout)

# Set the VQE solver
service = QiskitRuntimeService(channel="ibm_quantum")
optimizer=SLSQP(maxiter=1000)
with Session(service=service, backend="ibmq_qasm_simulator") as session:
    options= Options(optimization_level=3)
    options.resilience_level = 1
    options.execution.shots = 2048
    estimator=Estimator(session=session, options=options)
#session.close()  
vqe_solver = VQEUCCFactory(estimator=estimator,ansatz=ansatz,optimizer=optimizer)

# Run the VQE calculation using Ground_state_eigensolver  
# Perform the VQE calculation using the Ground state eigensolver
calc = GroundStateEigensolver(qubit_converter1, vqe_solver)
res = calc.solve(molecule)
print(res)

# Set the molecular strucutre using z-matrix
num_particles=(24,23)
co2 = 'O; C 1 1.18443330; O 2 {} 1 89.95941421'
distances = [x * 0.1 + 1.00 for x in range(30)]
energies = np.empty(len(distances))

for i, distance in enumerate(distances):
    # Initiate the PySCF driver
    driver = PySCFDriver(co2.format(distance), basis='sto3g')
    # Perform the setup calculation using PySCF
    pes_problem = ElectronicStructureProblem(driver)
    # Get the second quantized Hamiltonian
    second_q_op = pes_problem.second_q_ops()
    # Use Parity mapping to transform the Hamiltonian into qubitized form
    qubit_converter1 = QubitConverter(ParityMapper(), two_qubit_reduction=True)
    qubit_op1 = qubit_converter1.convert(second_q_op[0], num_particles)
    quantum_instance = QuantumInstance(backend = Aer.get_backend('statevector_simulator'))
    vqe_solver = VQEUCCFactory(quantum_instance)
    calc = GroundStateEigensolver(qubit_converter1, vqe_solver)
    res = calc.solve(pes_problem)
    hf_energy_list += [res.hartree_fock_energy] 
    uccsd_energy_list += [res.total_energies[0]]

#Print scan curve
plt.plot(distances,hf_energy_list,color='black',label='HF')
plt.plot(distances,uccsd_energy_list,color='red',label='UCCSD')
plt.title("Potential energy curve of silver(I) vs Carbon Dioxide molecule")
plt.xlabel("Scan")
plt.ylabel("Energy (hartrees)")
plt.legend()