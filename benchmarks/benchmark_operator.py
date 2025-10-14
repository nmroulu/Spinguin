"""
This script benchmarks the performance of creating Hilbert space
operators for varying spin systems. In this test, every spherical
tensor operator is created for each spin.

This benchmark is useful for comparing whether to use the dense
or sparse formalism for creating operators.

On a laptop with 11th gen. i5 processor and 16 GB ram, this bench-
mark takes a few minutes to run with `max_nspins=14`, with sparse
arrays becoming faster at ~10 spins.
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from spinguin._core.operators import op_prod

# Testing parameters
max_nspins = 14

# Create empty arrays for the testing results
avg_dense = np.empty(max_nspins, dtype=float)
avg_sparse = np.empty(max_nspins, dtype=float)

# Test with various spin system sizes
for nspins in range(1, max_nspins+1):

    print(f"Current number of spins: {nspins}")

    # Create the spin system
    spins = np.array([1/2 for _ in range(nspins)])

    # Define the operators to be tested
    opers = []
    for i in range(nspins):
        for j in range(4):
            op = np.zeros(nspins, dtype=int)
            op[i] = j
            opers.append(op)
    nopers = len(opers)

    # Total runtimes
    tot_dn = 0
    tot_sp = 0

    # Test with every product operator
    for op_def in opers:

        # Test with dense arrays
        ts = perf_counter()
        op_prod(op_def, spins, include_unit=True, sparse=False)
        te = perf_counter()
        tot_dn += te-ts

        # Test with sparse arrays
        ts = perf_counter()
        op_prod(op_def, spins, include_unit=True, sparse=True)
        te = perf_counter()
        tot_sp += te-ts

    # Save the results
    avg_dense[nspins-1] = tot_dn/nopers
    avg_sparse[nspins-1] = tot_sp/nopers

# Plot the results
nspins = np.linspace(1, max_nspins, num=max_nspins)
plt.plot(nspins, avg_dense, label="Dense")
plt.plot(nspins, avg_sparse, label="Sparse")
plt.xlabel("Number of spins")
plt.ylabel("Time (s)")
plt.legend()
plt.tight_layout()
plt.show()