"""
This script benchmarks the performance of creating Liouville space
superoperators for varying spin systems. In this test, every spherical
tensor operator is created for each spin.

On a laptop with 11th gen. i5 processor and 16 GB ram, this bench-
mark takes a minute to run with `max_nspins=14`, and `max_spin_order=4`.
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from spinguin.qm.superoperators import sop_prod
from spinguin.qm.basis import make_basis

# Testing parameters
max_nspins = 14
max_spin_order = 4

# Create empty arrays for the testing results
avg = np.empty((max_spin_order, max_nspins), dtype=float)

# Test with various maximum spin orders
for max_so in range(1, max_spin_order+1):

    print(f"Current spin order: {max_so}")

    # Test with various spin system sizes
    for nspins in range(max_so, max_nspins+1):

        print(f"Current number of spins: {nspins}")

        # Create the spin system
        spins = np.array([1/2 for _ in range(nspins)])
        basis = make_basis(spins, max_so)

        # Define the operators to be tested
        opers = []
        for i in range(nspins):
            for j in range(4):
                op = np.zeros(nspins, dtype=int)
                op[i] = j
                opers.append(op)
        nopers = len(opers)

        # Minimum and maximum runtimes
        tot_curr = 0

        # Test with every product operator
        for op_def in opers:

            # Test with sparse arrays
            ts = perf_counter()
            sop_prod(op_def, basis, spins, "comm", sparse=True)
            te = perf_counter()
            tot_curr += te-ts

        # Save the results
        avg[max_so-1, nspins-1] = tot_curr/nopers

# Plot the results
for max_so in range(1, max_spin_order+1):
    nspins = np.linspace(max_so, max_nspins, num=max_nspins-max_so)
    plt.plot(nspins, avg[max_so-1][max_so-1:max_nspins-1], label=f"Spin order: {max_so}")
plt.xlabel("Number of spins")
plt.ylabel("Time (s)")
plt.legend()
plt.tight_layout()
plt.show()