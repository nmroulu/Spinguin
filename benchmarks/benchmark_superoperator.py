"""
Benchmarks Liouville-space superoperator construction as a function of
spin-system size and basis-set truncation.

For each selected maximum spin order and number of spins, the benchmark
builds a Liouville-space basis and times the construction of all
superoperators whose operator definition has spin order below three.

On a laptop with an 11th generation i5 processor and 16 GB RAM, this
benchmark takes roughly two minutes with ``max_nspins = 14`` and
``max_spin_order = 4``.
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from spinguin._core._superoperators import sop_prod
from spinguin._core._basis import make_basis
from spinguin._core._utils import spin_order

# Define benchmark limits
max_nspins = 14
max_spin_order = 4

# Allocate array for average construction times
avg = np.empty((max_spin_order, max_nspins), dtype=float)

# Loop over maximum basis spin orders
for max_so in range(1, max_spin_order + 1):

    print(f"Current spin order: {max_so}")

    # Loop over spin-system sizes compatible with current truncation
    for nspins in range(max_so, max_nspins + 1):

        print(f"Current number of spins: {nspins}")

        # Build the spin-system description and Liouville-space basis
        spins = np.array([1/2 for _ in range(nspins)])
        basis = make_basis(spins, max_so)

        # Initialise the number of timed operators
        nopers = 0

        # Initialise cumulative runtime for current configuration
        tot_curr = 0

        # Time superoperator construction for selected basis operators
        for op_def in basis:

            # Restrict timed operators to at most two-spin correlations
            so = spin_order(op_def)
            if so < 3:

                # Measure superoperator construction wall time
                ts = perf_counter()
                sop_prod(op_def, basis, spins, "comm", sparse=True)
                te = perf_counter()
                nopers += 1
                tot_curr += te - ts

        # Store the mean construction time for this configuration
        avg[max_so - 1, nspins - 1] = tot_curr/nopers

# Plot average runtime as a function of spin-system size
for max_so in range(1, max_spin_order + 1):
    nspins = np.linspace(max_so, max_nspins, num=max_nspins - max_so)
    plt.plot(
        nspins,
        avg[max_so - 1][max_so - 1:max_nspins - 1],
        label=f"Spin order: {max_so}"
    )

# Finalise and show the benchmark figure
plt.xlabel("Number of spins")
plt.ylabel("Time (s)")
plt.legend()
plt.tight_layout()
plt.show()