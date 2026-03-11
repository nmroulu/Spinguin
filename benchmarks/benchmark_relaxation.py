"""
This script benchmarks the performance of creating a relaxation
superoperator using the Redfield theory. It uses a simple
algorithm for creating an arbitrary geometry for a molecule.

On a laptop with 11th gen. i5 processor and 16 GB ram, this bench-
mark takes a minute to run with `nspins=12` and `max_spin_order=3`.
"""

# Imports
import numpy as np
import spinguin as sg
sg.parameters.verbose = False

# Testing parameters
nspins = 12
max_spin_order = 3

# Set magnetic field
sg.parameters.magnetic_field = 1

# Create the spin system and build the basis
isotopes = np.array(["1H" for _ in range(nspins)])
spin_system = sg.SpinSystem(isotopes)
spin_system.basis.max_spin_order = max_spin_order
spin_system.basis.build()

# Create some relatively realistic geometry
xyz = []
for spin in range(nspins):
    theta = spin * 2.399963
    z = 1.5 * (spin / nspins - 0.5)
    r = 1.5 * np.sqrt(spin)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    xyz.append([x, y, z])
spin_system.xyz = xyz

# Get a distance array
connectors = spin_system.xyz[:, np.newaxis] - spin_system.xyz
distances = np.linalg.norm(connectors, axis=2)

# Create some chemical shifts
spin_system.chemical_shifts = [5 + 0.25*i*(-1)**i for i in range(nspins)]

# Create some J-couplings based on distances
J_couplings = np.zeros((nspins, nspins))
for i in range(nspins):
    for j in range(i):
        J_couplings[i,j] = 20 / distances[i,j]**3
spin_system.J_couplings = J_couplings

# Set relaxation settings
spin_system.relaxation.tau_c = 50e-12
spin_system.relaxation.theory = "redfield"

# Benchmark the Redfield superoperator
sg.relaxation(spin_system)