"""
This script simulates a simple example of SABRE hyperpolarization of pyridine.
The spin system consists of the hydride protons and one pyridine ligand (excluding 14N).
Chemical exchange and relaxation effects are not included in this simulation.

Execution time is approximately one minute on a laptop with an 11th-generation i5 processor.
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
from spinguin import SpinSystem, hamiltonian, propagator, singlet, measure

# Simulation settings
max_spin_order = 4  # Maximum spin order to consider
B0 = 7e-3  # Magnetic field strength in Tesla
dt = 1e-3  # Time step for simulation in seconds
N_steps = 1000  # Number of simulation steps

# Define isotopes
isotopes = np.array(['1H', '1H', '1H', '1H', '1H', '1H', '1H'])

# Define chemical shifts (in ppm)
chemical_shifts = np.array([-22.7, -22.7, 8.34, 8.34, 7.12, 7.12, 7.77])

# Define scalar couplings (in Hz)
J_couplings = np.array([
    [ 0,     0,      0,      0,      0,      0,      0],
    [-6.53,  0,      0,      0,      0,      0,      0],
    [ 0.00,  1.66,   0,      0,      0,      0,      0],
    [ 1.40,  0.00,  -0.06,   0,      0,      0,      0],
    [-0.09,	 0.35,	 6.03,	 0.14,	 0,      0,      0],
    [ 0.38, -0.13,	 0.09,	 5.93,	 0.06, 	 0,      0],
    [ 0.01,	 0.03,	 1.12,	-0.02,	 7.75,  -0.01, 	 0]
])

# Initialize the spin system
spin_system = SpinSystem(isotopes, chemical_shifts, J_couplings, max_spin_order=max_spin_order)

# Generate the Hamiltonian
H = hamiltonian(spin_system, B0)

# Generate the time propagator
P = propagator(dt, H)

# Create the initial state (singlet state for hydride spins)
rho = singlet(spin_system, 0, 1)

# Initialize an array to store magnetizations during evolution
magnetizations = np.empty((N_steps, isotopes.size), dtype=complex)

# Perform time evolution for the specified number of steps
for step in range(N_steps):

    # Propagate the system forward in time
    rho = P @ rho

    # Measure the magnetization for each spin
    for i in range(isotopes.size):
        magnetizations[step, i] = measure(spin_system, rho, 'I_z', i)

# Plot the magnetizations and display the results
for i in range(isotopes.size):
    plt.plot(np.real(magnetizations[:, i]), label=f"Spin {i+1}")
plt.legend(loc="upper right")
plt.xlabel("Time step")
plt.ylabel("Magnetization")
plt.title("SABRE Hyperpolarization of Pyridine")
plt.show()
plt.clf()
