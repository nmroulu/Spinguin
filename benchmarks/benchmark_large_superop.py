"""
Benchmarks the construction of a single Liouville-space superoperator for a
large spin-1/2 system.

The benchmark creates a restricted basis and constructs a left-multiplication
superoperator corresponding to a single-spin rank-1 spherical tensor operator
on the first spin.
"""

# Imports
import numpy as np
from time import perf_counter
from spinguin._core._basis import make_basis
from spinguin._core._superoperators import sop_prod
from spinguin._core._parameters import parameters

# Suppress status messages during benchmarking
parameters.verbose = False

# Use sparse matrices for superoperator construction
parameters.sparse_superoperator = True

print("Benchmark started.")

# Set benchmark parameters
nspins = 50
max_so = 3

# Create the spin system and the Liouville-space basis
spins = np.array([1/2 for _ in range(nspins)])
basis = make_basis(spins, max_so)

# Define the operator
op_def = np.zeros(nspins, dtype=int)

# Set the operator type for the first spin
op_def[0] = 2

# Time the superoperator construction
t_start = perf_counter()
sop_prod(op_def, basis, spins, "left")
t_end = perf_counter()

print(f"Constructing the superoperator took {t_end - t_start} seconds.")