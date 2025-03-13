"""
This function tests the speed of SciPy sparse matrix multiplications.
"""

# Imports
from scipy.sparse import random_array
from spinguin.la import sparse_dot
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
import os

# Densities to test
densities = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5])

# Maximum number of non-zeros
max_nnz = 1e6

# Data points
npoints = 30

# Go over the densities
for density in densities:

    # Get the maximum dimension to test
    max_dim = int(np.sqrt(max_nnz / density))

    # Dimensions to test
    dims = np.geomspace(start = 100, stop = max_dim, num=npoints, dtype=int)

    # Result arrays
    result_SciPy = np.empty(len(dims))
    result_custom = np.empty((len(dims)))

    # Test each dimension
    for i, dim in enumerate(dims):

        # Create a random array
        A = random_array((dim, dim), density=density, format='csc')

        # SciPy sparse implementation
        time_start_csr_csr = perf_counter()
        A @ A
        time_end_csr_csr = perf_counter()
        result_SciPy[i] = time_end_csr_csr - time_start_csr_csr

        # Custom sparse dot implementation
        time_start_custom = perf_counter()
        sparse_dot(A, A)
        time_end_custom = perf_counter()
        result_custom[i] = time_end_custom - time_start_custom

    # Plot the result
    plt.plot(dims, result_SciPy, label='SciPy')
    plt.plot(dims, result_custom, label='Custom.')
    plt.title(f"Density: {density}")
    plt.xlabel("Matrix dimension")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.tight_layout()
    script_path = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_path, 'results')
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f'density_{density}.png')
    plt.savefig(file_path)
    plt.clf()