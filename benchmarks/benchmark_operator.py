"""
This script benchmarks the performance of creating Hilbert space
operators for varying spin systems. In this test, every spherical
tensor operator is created for each spin.

This benchmark is useful for comparing whether to use the dense
or sparse formalism for creating operators.

On a laptop with 11th gen. i5 processor and 16 GB ram, this bench-
mark takes a few seconds to run with `max_nspins=12`, with sparse
arrays becoming faster at ~10 spins.
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from spinguin.qm.operators import op_prod

# Testing parameters
max_nspins = 12

# Create empty arrays for the testing results
min_dense = np.empty(max_nspins, dtype=float)
max_dense = np.empty(max_nspins, dtype=float)
avg_dense = np.empty(max_nspins, dtype=float)
min_sparse = np.empty(max_nspins, dtype=float)
max_sparse = np.empty(max_nspins, dtype=float)
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

    # Minimum and maximum runtimes
    min_dn = np.inf
    max_dn = 0
    tot_dn = 0
    min_sp = np.inf
    max_sp = 0
    tot_sp = 0

    # Test with every product operator
    for op_def in opers:
        op_def = np.array(op_def)

        # Test with dense arrays
        ts = perf_counter()
        op_prod(op_def, spins, include_unit=True, sparse=False)
        te = perf_counter()
        t = te-ts
        tot_dn += t
        if t < min_dn:
            min_dn = t
        if t > max_dn:
            max_dn = t

        # Test with sparse arrays
        ts = perf_counter()
        op_prod(op_def, spins, include_unit=True, sparse=True)
        te = perf_counter()
        t = te-ts
        tot_sp += t
        if t < min_sp:
            min_sp = t
        if t > max_sp:
            max_sp = t

    # Save the results
    min_dense[nspins-1] = min_dn
    max_dense[nspins-1] = max_dn
    avg_dense[nspins-1] = tot_dn/nopers
    min_sparse[nspins-1] = min_sp
    max_sparse[nspins-1] = max_sp
    avg_sparse[nspins-1] = tot_sp/nopers

# Plot the results
nspins = np.linspace(1, nspins, num=nspins)
plt.plot(nspins, min_dense, label="Dense: MIN")
plt.plot(nspins, max_dense, label="Dense: MAX")
plt.plot(nspins, avg_dense, label="Dense: AVG")
plt.plot(nspins, min_sparse, label="Sparse: MIN")
plt.plot(nspins, max_sparse, label="Sparse: MAX")
plt.plot(nspins, avg_sparse, label="Sparse: AVG")
plt.xlabel("Number of spins")
plt.ylabel("Time (s)")
plt.legend()
plt.show()