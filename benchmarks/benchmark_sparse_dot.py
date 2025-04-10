"""
This script benchmarks the performance of SciPy's sparse matrix multiplication
against a custom sparse dot implementation for various matrix densities.
"""

# Imports
from scipy.sparse import random_array  # Import for generating random sparse matrices
from spinguin import sparse_dot  # Custom sparse dot implementation
from time import perf_counter  # For measuring execution time
import numpy as np
import matplotlib.pyplot as plt
import os

# Densities to test (fraction of non-zero elements in the matrix)
densities = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5])

# Maximum number of non-zero elements allowed in the matrix
max_nnz = 1e6

# Number of matrix dimensions to test for each density
npoints = 30

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
        A = random_array((dim, dim), density=density, format='csc')

        # Measure execution time for SciPy sparse matrix multiplication
        time_start_csr_csr = perf_counter()
        A @ A  # Perform matrix multiplication
        time_end_csr_csr = perf_counter()
        result_SciPy[i] = time_end_csr_csr - time_start_csr_csr

        # Measure execution time for the custom sparse dot implementation
        time_start_custom = perf_counter()
        sparse_dot(A, A)  # Perform matrix multiplication
        time_end_custom = perf_counter()
        result_custom[i] = time_end_custom - time_start_custom

    # Plot the benchmark results for the current density
    plt.plot(dims, result_SciPy, label='SciPy')  # SciPy results
    plt.plot(dims, result_custom, label='Custom')  # Custom implementation results
    plt.title(f"Density: {density}")  # Title indicating the current density
    plt.xlabel("Matrix Dimension")  # X-axis label
    plt.ylabel("Time (s)")  # Y-axis label
    plt.legend()  # Add legend to distinguish between implementations
    plt.tight_layout()  # Adjust layout for better readability

    # Save the plot to the 'results' folder
    script_path = os.path.dirname(os.path.abspath(__file__))  # Get script directory
    folder_path = os.path.join(script_path, 'results')  # Path to 'results' folder
    os.makedirs(folder_path, exist_ok=True)  # Create folder if it doesn't exist
    file_path = os.path.join(folder_path, f'density_{density}.png')  # File path for plot
    plt.savefig(file_path)  # Save the plot as a PNG file
    plt.clf()  # Clear the figure for the next plot