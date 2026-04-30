"""
Benchmarks the construction of a Redfield relaxation superoperator for an
artificial proton many-spin system.

The benchmark builds a simple helical-like geometry, assigns approximate
chemical shifts and distance-based scalar couplings, and then evaluates
the relaxation superoperator.

On a laptop with 11th gen. i5 processor and 16 GB RAM, this benchmark takes
about one minute with ``nspins = 12`` and ``max_spin_order = 3``.
"""

# Imports
import numpy as np
from time import perf_counter
import spinguin as sg

# Suppress verbose output during benchmarking
sg.parameters.verbose = False

# Define benchmark parameters
nspins = 12
max_spin_order = 3

# Set magnetic field strength in Tesla
sg.parameters.magnetic_field = 1

# Create the spin system and construct the Liouville-space basis
isotopes = np.array(["1H" for _ in range(nspins)])
spin_system = sg.SpinSystem(isotopes)
spin_system.basis.max_spin_order = max_spin_order
spin_system.basis.build()

# Build a simple molecular geometry with distributed spin positions
xyz = []
for spin in range(nspins):
    theta = spin * 2.399963
    z = 1.5 * (spin / nspins - 0.5)
    r = 1.5 * np.sqrt(spin)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    xyz.append([x, y, z])
spin_system.xyz = xyz

# Compute pairwise inter-spin distances
connectors = spin_system.xyz[:, np.newaxis] - spin_system.xyz
distances = np.linalg.norm(connectors, axis=2)

# Assign simple alternating chemical shifts in ppm
spin_system.chemical_shifts = [5 + 0.25 * i * (-1) ** i for i in range(nspins)]

# Assign scalar couplings using an arbitrary distance-cubed scaling
J_couplings = np.zeros((nspins, nspins))
for i in range(nspins):
    for j in range(i):
        J_couplings[i, j] = 20 / distances[i, j] ** 3
spin_system.J_couplings = J_couplings

# Set Redfield relaxation theory parameters
spin_system.relaxation.tau_c = 50e-12
spin_system.relaxation.theory = "redfield"

# Benchmark relaxation superoperator construction
t_start = perf_counter()
sg.relaxation(spin_system)
t_end = perf_counter()

print(f"Constructing the relaxation superoperator took {t_end - t_start} seconds.")