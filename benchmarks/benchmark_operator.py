"""
Benchmarks the performance of constructing Hilbert-space operators for
varying spin-system sizes. All spins are taken as spin-1/2, and for each
spin all four single-spin spherical tensor operators are generated
using both the dense and sparse back-ends. The operator index is
given by N = l^2 + l - q, where l is the rank and q is the projection
of the spherical tensor operator.

This benchmark is useful for determining the crossover point in
spin-system size at which the sparse-array formalism becomes more efficient
than the dense formalism for constructing operators.

On a laptop with 11th gen. i5 processor and 16 GB RAM, this benchmark
takes a few minutes to run with ``max_nspins = 14``, with sparse
arrays becoming faster at approximately 10 spins.
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from spinguin._core._operators import op_prod
from spinguin._core._parameters import parameters

# Suppress verbose output during benchmarking
parameters.verbose = False

# Maximum number of spins to test
max_nspins = 14

# Initialise empty arrays for the timing results
avg_dense = np.empty(max_nspins, dtype=float)
avg_sparse = np.empty(max_nspins, dtype=float)

# Test with various spin system sizes
for nspins in range(1, max_nspins + 1):

    print(f"Current number of spins: {nspins}")

    # Create the spin system (all spins are spin-1/2)
    spins = np.array([1/2 for _ in range(nspins)])

    # Construct operator definitions for all single-spin spherical tensor
    # operators; for spin-1/2 there are four operators per spin (N = 0,...,3)
    oper_defs = []
    for i in range(nspins):
        for j in range(4):
            oper_def = np.zeros(nspins, dtype=int)
            oper_def[i] = j
            oper_defs.append(oper_def)
    nopers = len(oper_defs)

    # Initialise cumulative runtimes for the dense and sparse back-ends
    tot_dense = 0
    tot_sparse = 0

    # Measure construction time for every operator definition
    for op_def in oper_defs:

        # Measure construction time using the dense back-end
        parameters.sparse_operator = False
        ts = perf_counter()
        op_prod(op_def, spins, include_unit=True)
        te = perf_counter()
        tot_dense += te - ts

        # Measure construction time using the sparse back-end
        parameters.sparse_operator = True
        ts = perf_counter()
        op_prod(op_def, spins, include_unit=True)
        te = perf_counter()
        tot_sparse += te - ts

    # Compute and store the mean construction time for the current system
    avg_dense[nspins - 1] = tot_dense/nopers
    avg_sparse[nspins - 1] = tot_sparse/nopers

# Plot the timing results as a function of spin system size
nspins = np.linspace(1, max_nspins, num=max_nspins)
plt.plot(nspins, avg_dense, label="Dense")
plt.plot(nspins, avg_sparse, label="Sparse")
plt.xlabel("Number of spins")
plt.ylabel("Time (s)")
plt.legend()
plt.tight_layout()
plt.show()