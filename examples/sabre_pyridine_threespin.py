"""
This is a simple example of a SABRE simulation of pyridine.
The spin system consists of the hydride protons and one ligand proton.
Chemical exchange and relaxation are not simulated.

Takes a few seconds to run on a laptop with 11th generation i5 processor.
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
from hyppy.spin_system import SpinSystem
from hyppy.hamiltonian import hamiltonian
from hyppy.propagation import propagator
from hyppy.states import singlet, measure

# Simulation settings
magnetic_field = 7e-3
time_step = 1e-3
nsteps = 1000

# Assign isotopes
isotopes = np.array(['1H', '1H', '1H'])

# Assign chemical shifts
chemical_shifts = np.array([-22.7, -22.7, 8.34])

# Assign scalar couplings
scalar_couplings = np.array([\
    [ 0,     0,      0],
    [-6.53,  0,      0],
    [ 0.00,  1.66,   0]
])

# Initialize the spin system
spin_system = SpinSystem(isotopes, chemical_shifts, scalar_couplings)

# Make the Hamiltonian
H = hamiltonian(spin_system, magnetic_field)

# Make the time propagator
P = propagator(time_step, H)

# Create the initial state (singlet for hydride spins)
rho = singlet(spin_system, 0, 1)

# Create an array for storing the magnetizations during evolution
magnetizations = np.empty((nsteps, isotopes.size), dtype=complex)

# Evolve for the number of steps
for step in range(nsteps):

    # Propagate the system forward in time
    rho = P @ rho

    # Measure the magnetization for each spin
    for i in range(isotopes.size):
        magnetizations[step, i] = measure(spin_system, rho, 'I_z', i)

# Plot the magnetizations and show the result
for i in range(isotopes.size):
    plt.plot(np.real(magnetizations[:,i]), label=f"Spin {i+1}")
plt.legend(loc="upper right")
plt.xlabel("Time step")
plt.ylabel("Magnetization")
plt.title("SABRE-hyperpolarization of Pyridine")
plt.show()
plt.clf()
