"""
Benchmarks sparse matrix multiplication using SciPy against Spinguin's custom
sparse dot implementation for a range of matrix densities.

For each density, random square sparse matrices are generated over a range of
dimensions and both methods are timed using repeated multiplications.

NOTE: This benchmark takes several minutes to run.
"""

# Imports
import os
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import random_array
from spinguin._core._la import custom_dot

# Densities to test (fraction of non-zero elements in the matrix)
densities = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5])

# Maximum number of non-zero elements allowed in the matrix
max_nnz = 1e6

# Number of matrix dimensions to test for each density
npoints = 30

# Number of repeated multiplications per matrix size
nrounds = 100

# Create the output directory for benchmark figures
script_path = os.path.dirname(os.path.abspath(__file__))
output_directory = os.path.join(script_path, 'results')
os.makedirs(output_directory, exist_ok=True)

# Iterate over the specified densities
for density in densities:

    # Determine the largest test dimension allowed by the nnz limit
    max_dim = int(np.sqrt(max_nnz / density))

    # Generate logarithmically spaced matrix dimensions for this density
    matrix_dimensions = np.geomspace(
        start=100,
        stop=max_dim,
        num=npoints,
        dtype=int
    )

    # Initialise arrays for SciPy and custom timing results
    result_scipy = np.empty(len(matrix_dimensions), dtype=float)
    result_custom = np.empty(len(matrix_dimensions), dtype=float)

    # Test each matrix dimension
    for i, dim in enumerate(matrix_dimensions):

        # Construct one random matrix in both CSR and CSC formats
        matrix_csr = random_array(
            shape=(dim, dim),
            density=density,
            format='csr',
            dtype=complex
        )
        matrix_csc = matrix_csr.tocsc()

        # Measure execution time for SciPy sparse matrix multiplication
        time_start_scipy = perf_counter()
        for _ in range(nrounds):
            matrix_csr @ matrix_csr
        time_end_scipy = perf_counter()
        result_scipy[i] = (time_end_scipy - time_start_scipy)/nrounds

        # Measure execution time for the custom sparse dot implementation
        time_start_custom = perf_counter()
        for _ in range(nrounds):
            custom_dot(matrix_csc, matrix_csc, zero_value=0)
        time_end_custom = perf_counter()
        result_custom[i] = (time_end_custom - time_start_custom)/nrounds

    # Plot the benchmark results for the current density
    plt.plot(matrix_dimensions, result_scipy, label='SciPy')
    plt.plot(matrix_dimensions, result_custom, label='Custom')
    plt.title(f"Density: {density}")
    plt.xlabel("Matrix Dimension")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.tight_layout()

    # Save the plot for the current density
    file_path = os.path.join(output_directory, f'density_{density}.png')
    plt.savefig(file_path)
    plt.clf()