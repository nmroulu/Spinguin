"""
This module provides a Cython wrapper for the sparse matrix multiplication
algorithm written in C++.
"""

import numpy as np
cimport numpy as np
from scipy.sparse import csc_array

# Define the C++ functions
cdef extern from "c_sparse_dot.hpp" nogil:
    cdef void c_sparse_dot_indptr[I, T](
        T* A_data, I* A_indices, I* A_indptr, I A_nrows,
        T* B_data, I* B_indices, I* B_indptr, I B_ncols,
        I* C_indptr, np.float64_t zero_value
    )
    cdef void c_sparse_dot[I, T](
        T* A_data, I* A_indices, I* A_indptr, I A_nrows,
        T* B_data, I* B_indices, I* B_indptr, I B_ncols,
        T* C_data, I* C_indices, I* C_indptr,
        np.float64_t zero_value
    )

# Possible types for index and pointer arrays
ctypedef fused IType:
    np.int32_t
    np.int64_t

# Possible types for data arrays
ctypedef fused TType:
    np.int32_t
    np.int64_t
    np.float64_t
    np.complex128_t

def _cy_sparse_dot_indptr(
    TType[::1] A_data, IType[::1] A_indices, IType[::1] A_indptr, IType A_nrows,
    TType[::1] B_data, IType[::1] B_indices, IType[::1] B_indptr, IType B_ncols,
    IType[::1] C_indptr, np.float64_t zero_value
) -> None:
    """
    Cython wrapper for the C++ function that calculates the index pointer
    array for the resulting matrix in a matrix multiplication. Modification
    of the index pointer array happens inplace.
    """
    # Release the GIL and find the index pointer array
    with nogil:
        c_sparse_dot_indptr(
            &A_data[0], &A_indices[0], &A_indptr[0], A_nrows,
            &B_data[0], &B_indices[0], &B_indptr[0], B_ncols,
            &C_indptr[0], zero_value)

def _cy_sparse_dot(
    TType[::1] A_data, IType[::1] A_indices, IType[::1] A_indptr, IType A_nrows,
    TType[::1] B_data, IType[::1] B_indices, IType[::1] B_indptr, IType B_ncols,
    TType[::1] C_data, IType[::1] C_indices, IType[::1] C_indptr,
    np.float64_t zero_value
) -> None:
    """
    Cython wrapper for the C++ matrix multiplication function. Modification of
    the result matrix happens inplace.
    """
    # Release the GIL and perform the matrix multiplication
    with nogil:
        c_sparse_dot(            
            &A_data[0], &A_indices[0], &A_indptr[0], A_nrows,
            &B_data[0], &B_indices[0], &B_indptr[0], B_ncols,
            &C_data[0], &C_indices[0], &C_indptr[0], zero_value)

def sparse_dot(A: csc_array, B: csc_array, zero_value: float) -> csc_array:
    """
    Custom matrix multiplication for SciPy CSC matrices.

    Parameters
    ----------
    A : csc_array
        Matrix A.
    B : csc_array
        Matrix B.
    zero_value : float
        Threshold below which a number is considered zero in the result matrix.

    Returns
    -------
    C : csc_array
        Result of the matrix multiplication C = A @ B.
    """
    # Accept only CSC arrays
    if not (isinstance(A, csc_array) and isinstance(B, csc_array)):
        raise ValueError("The input arrays must be of type CSC.")

    # Obtain the dimensions of the result matrix
    A_nrows = A.shape[0]
    B_ncols = B.shape[1]

    # Shortcut for empty matrix
    if A.nnz == 0 or B.nnz == 0:
        C = csc_array((A_nrows, B_ncols))
        return C

    # Find out smallest common data types
    dtype_AI = np.promote_types(A.indices.dtype, A.indptr.dtype)
    dtype_BI = np.promote_types(B.indices.dtype, B.indptr.dtype)
    dtype_I = np.promote_types(dtype_AI, dtype_BI)
    dtype_T = np.promote_types(A.data.dtype, B.data.dtype)

    # Perform type conversions
    A_data = A.data.astype(dtype_T, copy=False)
    A_indices = A.indices.astype(dtype_I, copy=False)
    A_indptr = A.indptr.astype(dtype_I, copy=False)
    B_data = B.data.astype(dtype_T, copy=False)
    B_indices = B.indices.astype(dtype_I, copy=False)
    B_indptr = B.indptr.astype(dtype_I, copy=False)

    # Create the index pointer array for result matrix
    C_indptr = np.zeros(B_ncols+1, dtype=dtype_I)

    # Calculate the index pointer array for result matrix
    _cy_sparse_dot_indptr(
        A_data, A_indices, A_indptr, A_nrows,
        B_data, B_indices, B_indptr, B_ncols,
        C_indptr, zero_value
    )

    # Obtain the true index pointer array by taking cumulative sum
    C_indptr = np.cumsum(C_indptr)
    
    # Find the number of non-zeros
    nnz = C_indptr[B_ncols]

    # Special case for empty matrix
    if nnz == 0:
        C = csc_array((A_nrows, B_ncols))
        return C

    # Maximum integer for 32 bits
    max_32 = 2**31 - 1
        
    # In case of overflow, change to 64 bits
    if nnz > max_32 and dtype_I == np.int32:
        dtype_I = np.int64
        A_indices = A_indices.astype(dtype_I)
        A_indptr = A_indptr.astype(dtype_I)
        B_indices = B_indices.astype(dtype_I)
        B_indptr = B_indptr.astype(dtype_I)
    
    # Otherwise make sure that the data types match
    else:
        C_indptr = C_indptr.astype(dtype_I, copy=False)

    # Create the result array
    C_data = np.zeros(nnz, dtype=dtype_T)
    C_indices = np.zeros(nnz, dtype=dtype_I)
    
    # Perform the matrix multiplication (modifies C inplace)
    _cy_sparse_dot(
        A_data, A_indices, A_indptr, A_nrows,
        B_data, B_indices, B_indptr, B_ncols,
        C_data, C_indices, C_indptr,
        zero_value
    )

    # Construct the SciPy CSC array
    C = csc_array((C_data, C_indices, C_indptr), shape=(A_nrows, B_ncols))

    return C