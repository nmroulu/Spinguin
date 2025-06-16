"""
This script benchmarks the performance of SciPy's sparse matrix multiplication
against a custom sparse dot implementation for various matrix densities.

NOTE: Takes several minutes to run!
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.sparse import random_array
from spinguin.core.la import custom_dot
from time import perf_counter

# Densities to test (fraction of non-zero elements in the matrix)
densities = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5])

# Maximum number of non-zero elements allowed in the matrix
max_nnz = 1e6

# Number of matrix dimensions to test for each density
npoints = 30

# Averaging
nrounds = 100

# Iterate over the specified densities
for density in densities:

    # Calculate the maximum matrix dimension based on the density and max_nnz
    max_dim = int(np.sqrt(max_nnz / density))

    # Generate a range of matrix dimensions to test (logarithmically spaced)
    dims = np.geomspace(start=100, stop=max_dim, num=npoints, dtype=int)

    # Arrays to store benchmark results for SciPy and custom implementations
    result_SciPy = np.empty(len(dims))
    result_custom = np.empty(len(dims))

    # Test each matrix dimension
    for i, dim in enumerate(dims):

        # Create a random sparse matrix with the given dimension and density
        A_CSR = random_array((dim, dim), density=density, format='csr',
                             dtype=complex)
        A_CSC = A_CSR.tocsc()

        # Measure execution time for SciPy sparse matrix multiplication
        time_start_SciPy = perf_counter()
        for _ in range(nrounds):
            A_CSR @ A_CSR
        time_end_SciPy = perf_counter()
        result_SciPy[i] = (time_end_SciPy - time_start_SciPy)/nrounds

        # Measure execution time for the custom sparse dot implementation
        time_start_custom = perf_counter()
        for _ in range(nrounds):
            custom_dot(A_CSC, A_CSC, zero_value=0)
        time_end_custom = perf_counter()
        result_custom[i] = (time_end_custom - time_start_custom)/nrounds

    # Plot the benchmark results for the current density
    plt.plot(dims, result_SciPy, label='SciPy')
    plt.plot(dims, result_custom, label='Custom')
    plt.title(f"Density: {density}")
    plt.xlabel("Matrix Dimension")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.tight_layout()

    # Save the plot to the 'results' folder
    script_path = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_path, 'results')
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f'density_{density}.png')
    plt.savefig(file_path)
    plt.clf()