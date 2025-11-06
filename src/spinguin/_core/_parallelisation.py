"""
This module contains functions that allow sharing SciPy sparse arrays across
Python interpreters, allowing the use of parallelisation.
"""
# Imports
import numpy as np
import scipy.sparse as sp
from multiprocessing.shared_memory import SharedMemory

def read_shared_sparse(A_shared: dict[str, str | np.dtype | tuple[int]]) -> \
    tuple[sp.csc_array, tuple[SharedMemory, SharedMemory, SharedMemory]]:
    """
    Reads a shared memory representation of a sparse CSC array and reconstructs
    it.

    Parameters
    ----------
    A_shared : dict
        Dictionary containing shared memory names and metadata for the sparse
        array's data, indices, and indptr, along with their shapes and dtypes.

    Returns
    -------
    A : csc_array
        Sparse array reconstructed from the shared memory.
    A_shm : tuple
        Tuple containing the shared memory objects for the sparse array's data,
        indices, and indptr.
    """
    # Parse the dictionary
    A_data_shm_name = A_shared['A_data_shm_name']
    A_indices_shm_name = A_shared['A_indices_shm_name']
    A_indptr_shm_name = A_shared['A_indptr_shm_name']
    A_data_shape = A_shared['A_data_shape']
    A_indices_shape = A_shared['A_indices_shape']
    A_indptr_shape = A_shared['A_indptr_shape']
    A_data_dtype = A_shared['A_data_dtype']
    A_indices_dtype = A_shared['A_indices_dtype']
    A_indptr_dtype = A_shared['A_indptr_dtype']
    A_shape = A_shared['A_shape']

    # Obtain the shared memories
    A_data_shm = SharedMemory(name=A_data_shm_name)
    A_indices_shm = SharedMemory(name=A_indices_shm_name)
    A_indptr_shm = SharedMemory(name=A_indptr_shm_name)

    # Obtain the previously shared array
    A_data = np.ndarray(shape=A_data_shape, dtype=A_data_dtype,
                        buffer=A_data_shm.buf)
    A_indices = np.ndarray(shape=A_indices_shape, dtype=A_indices_dtype,
                           buffer=A_indices_shm.buf)
    A_indptr = np.ndarray(shape=A_indptr_shape, dtype=A_indptr_dtype,
                          buffer=A_indptr_shm.buf)

    # Create the sparse array
    A = sp.csc_array((A_data, A_indices, A_indptr), shape=A_shape, copy=False)

    # Combine the shared memories into one tuple
    A_shm = (A_data_shm, A_indices_shm, A_indptr_shm)

    return A, A_shm

def write_shared_sparse(A: sp.csc_array) -> tuple[
    dict[str, str | np.dtype | tuple[int]],
    tuple[SharedMemory, SharedMemory, SharedMemory]]:
    """
    Creates a shared memory representation of a sparse CSC array.

    Parameters
    ----------
    A : csc_array
        Sparse array to be shared.

    Returns
    -------
    A_shared : dict
        Dictionary containing shared memory names and metadata for the sparse
        array's data, indices, and indptr, along with their shapes and dtypes.
    A_shm : tuple
        Tuple containing the shared memory objects for the sparse array's data,
        indices, and indptr.
    """
    # Create a shared memory of the sparse array
    A_data_shm = SharedMemory(create=True, size=A.data.nbytes)
    A_indices_shm = SharedMemory(create=True, size=A.indices.nbytes)
    A_indptr_shm = SharedMemory(create=True, size=A.indptr.nbytes)
    A_data_shared = np.ndarray(A.data.shape, dtype=A.data.dtype,
                               buffer=A_data_shm.buf)
    A_indices_shared = np.ndarray(A.indices.shape, dtype=A.indices.dtype,
                                  buffer=A_indices_shm.buf)
    A_indptr_shared = np.ndarray(A.indptr.shape, dtype=A.indptr.dtype,
                                 buffer=A_indptr_shm.buf)
    A_data_shared[:] = A.data[:]
    A_indices_shared[:] = A.indices[:]
    A_indptr_shared[:] = A.indptr[:]

    # Save the information of the memory to a dictionary
    A_shared = {
        'A_data_shm_name' : A_data_shm.name,
        'A_indices_shm_name' : A_indices_shm.name,
        'A_indptr_shm_name' : A_indptr_shm.name,
        'A_data_shape' : A.data.shape,
        'A_indices_shape' : A.indices.shape,
        'A_indptr_shape' : A.indptr.shape,
        'A_data_dtype' : A.data.dtype,
        'A_indices_dtype' : A.indices.dtype,
        'A_indptr_dtype' : A.indptr.dtype,
        'A_shape' : A.shape
    }

    # Combine the shared memories into one tuple
    A_shm = (A_data_shm, A_indices_shm, A_indptr_shm)

    return A_shared, A_shm