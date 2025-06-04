"""
A test that constructs a superoperator for a large spin system.
"""

# Imports
import numpy as np
from time import perf_counter
from spinguin.core.basis import make_basis
from spinguin.core.superoperators import sop_prod

print("Test started.")

# Test settings
nspins = 50
max_so = 3

# Create spin system
spins = np.array([1/2 for _ in range(nspins)])
basis = make_basis(spins, max_so)

# Create test superoperator
op_def = np.zeros(nspins, dtype=int)
op_def[0] = 2
ts = perf_counter()
sop_prod(op_def, basis, spins, "left", sparse=True)
te = perf_counter()

print(f"Constructing the superoperator took {te-ts} seconds.")