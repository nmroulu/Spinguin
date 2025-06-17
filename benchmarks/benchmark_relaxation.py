"""
This script benchmarks the performance of creating a relaxation
superoperator using the Redfield theory. It uses a simple
algorithm for creating an arbitrary geometry for a molecule.

On a laptop with 11th gen. i5 processor and 16 GB ram, this bench-
mark takes a minute to run with `nspins=12` and `max_spin_order=3`.
"""

# Imports
import numpy as np
from spinguin.core.nmr_isotopes import ISOTOPES
from spinguin.core.hamiltonian import sop_H
from spinguin.core.relaxation import sop_R_redfield
from spinguin.core.basis import make_basis

# Testing parameters
nspins = 12
max_spin_order = 3

# Set magnetic field
B = 1

# Set correlation time
tau_c = 50e-12

# Create the spin system
spins = np.array([1/2 for _ in range(nspins)])
basis = make_basis(spins, max_spin_order)

# Obtain the gyromagnetic ratios
y1H = 2*np.pi * ISOTOPES['1H'][1] * 1e6
gammas = np.array([y1H for _ in range(nspins)])
quad = np.array([0 for _ in range(nspins)])

# Create some geometry
xyz = []
for spin in range(nspins):
    theta = spin * 2.399963
    z = 1.5 * (spin / nspins - 0.5)
    r = 1.5 * np.sqrt(spin)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    xyz.append([x, y, z])
xyz = np.array(xyz)

# Get a distance array
connectors = xyz[:, np.newaxis] - xyz
distances = np.linalg.norm(connectors, axis=2)

# Create some chemical shifts
chemical_shifts = np.array([5 + 0.25*i*(-1)**i for i in range(nspins)])

# Create some J-couplings based on distances
J_couplings = np.zeros((nspins, nspins))
for i in range(nspins):
    for j in range(i):
        J_couplings[i,j] = 20 / distances[i,j]**3

# Create the Hamiltonian superoperator
H = sop_H(basis, spins, gammas, B, chemical_shifts, J_couplings)

# Benchmark the Redfield superoperator
sop_R_redfield(basis = basis,
               sop_H = H,
               tau_c = tau_c,
               spins = spins,
               B = B,
               gammas = gammas,
               xyz = xyz)

