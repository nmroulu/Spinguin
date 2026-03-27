"""
sparse_dot.pyx

Provides Cython wrappers for the C++ sparse matrix multiplication backend.
The public entry point, `sparse_dot`, multiplies two SciPy CSC arrays while
delegating the low-level sparse traversal to compiled code.
"""

# Imports
import numpy as np
cimport numpy as np
cimport cython
from scipy.sparse import csc_array

# Define the C++ backend functions
cdef extern from "_c_sparse_dot.hpp" nogil:
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

# Define the admissible integer types for sparse structure arrays
ctypedef fused IType:
    np.int32_t
    np.int64_t

# Define the admissible numerical types for sparse data arrays
ctypedef fused TType:
    np.int32_t
    np.int64_t
    np.float64_t
    np.complex128_t

@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def _cy_sparse_dot_indptr(
    TType[::1] A_data,
    IType[::1] A_indices,
    IType[::1] A_indptr,
    IType A_nrows,
    TType[::1] B_data,
    IType[::1] B_indices,
    IType[::1] B_indptr,
    IType B_ncols,
    IType[::1] C_indptr,
    np.float64_t zero_value,
) -> None:
    """
    Populate the column-pointer counts for the sparse product.

    Parameters
    ----------
    A_data : memoryview
        Non-zero values of the left CSC matrix.
    A_indices : memoryview
        Row indices of the left CSC matrix.
    A_indptr : memoryview
        Column-pointer array of the left CSC matrix.
    A_nrows : int
        Number of rows in the left matrix.
    B_data : memoryview
        Non-zero values of the right CSC matrix.
    B_indices : memoryview
        Row indices of the right CSC matrix.
    B_indptr : memoryview
        Column-pointer array of the right CSC matrix.
    B_ncols : int
        Number of columns in the right matrix.
    C_indptr : memoryview
        Output array that is filled in place with per-column non-zero counts.
    zero_value : float
        Threshold below which values are discarded as numerical zeros.

    Returns
    -------
    None
        The output is written in place to `C_indptr`.
    """

    # Release the GIL and count the non-zeros in each output column.
    with nogil:
        c_sparse_dot_indptr(
            &A_data[0], &A_indices[0], &A_indptr[0], A_nrows,
            &B_data[0], &B_indices[0], &B_indptr[0], B_ncols,
            &C_indptr[0], zero_value,
        )

@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def _cy_sparse_dot(
    TType[::1] A_data,
    IType[::1] A_indices,
    IType[::1] A_indptr,
    IType A_nrows,
    TType[::1] B_data,
    IType[::1] B_indices,
    IType[::1] B_indptr,
    IType B_ncols,
    TType[::1] C_data,
    IType[::1] C_indices,
    IType[::1] C_indptr,
    np.float64_t zero_value,
) -> None:
    """
    Compute the sparse matrix product into preallocated CSC buffers.

    Parameters
    ----------
    A_data : memoryview
        Non-zero values of the left CSC matrix.
    A_indices : memoryview
        Row indices of the left CSC matrix.
    A_indptr : memoryview
        Column-pointer array of the left CSC matrix.
    A_nrows : int
        Number of rows in the left matrix.
    B_data : memoryview
        Non-zero values of the right CSC matrix.
    B_indices : memoryview
        Row indices of the right CSC matrix.
    B_indptr : memoryview
        Column-pointer array of the right CSC matrix.
    B_ncols : int
        Number of columns in the right matrix.
    C_data : memoryview
        Output buffer for the non-zero values of the product.
    C_indices : memoryview
        Output buffer for the row indices of the product.
    C_indptr : memoryview
        Column-pointer array of the product matrix.
    zero_value : float
        Threshold below which values are discarded as numerical zeros.

    Returns
    -------
    None
        The result is written in place to `C_data` and `C_indices`.
    """

    # Release the GIL and evaluate the sparse matrix product.
    with nogil:
        c_sparse_dot(
            &A_data[0], &A_indices[0], &A_indptr[0], A_nrows,
            &B_data[0], &B_indices[0], &B_indptr[0], B_ncols,
            &C_data[0], &C_indices[0], &C_indptr[0], zero_value,
        )

def sparse_dot(
    A: csc_array,
    B: csc_array,
    zero_value: float,
) -> csc_array:
    """
    Multiply two SciPy CSC arrays using the compiled sparse backend.

    Parameters
    ----------
    A : csc_array
        Left matrix in compressed sparse column format.
    B : csc_array
        Right matrix in compressed sparse column format.
    zero_value : float
        Threshold below which a number is considered zero in the result
        matrix.

    Returns
    -------
    C : csc_array
        Sparse product `C = A @ B` in compressed sparse column format.

    Raises
    ------
    ValueError
        Raised if either input matrix is not a SciPy `csc_array`.
    """

    # Accept only SciPy CSC arrays as inputs.
    if not (isinstance(A, csc_array) and isinstance(B, csc_array)):
        raise ValueError("The input arrays must be of type CSC.")

    # Obtain the dimensions of the product matrix.
    A_nrows = A.shape[0]
    B_ncols = B.shape[1]

    # Return immediately if either input matrix is empty.
    if A.nnz == 0 or B.nnz == 0:
        return csc_array((A_nrows, B_ncols))

    # Promote index types and numerical data types independently.
    dtype_AI = np.promote_types(A.indices.dtype, A.indptr.dtype)
    dtype_BI = np.promote_types(B.indices.dtype, B.indptr.dtype)
    dtype_I = np.promote_types(dtype_AI, dtype_BI)
    dtype_T = np.promote_types(A.data.dtype, B.data.dtype)

    # Convert the sparse storage arrays to the promoted data types.
    A_data = A.data.astype(dtype_T, copy=False)
    A_indices = A.indices.astype(dtype_I, copy=False)
    A_indptr = A.indptr.astype(dtype_I, copy=False)
    B_data = B.data.astype(dtype_T, copy=False)
    B_indices = B.indices.astype(dtype_I, copy=False)
    B_indptr = B.indptr.astype(dtype_I, copy=False)

    # Allocate storage for the output column counts.
    C_indptr = np.zeros(B_ncols + 1, dtype=dtype_I)

    # Count the number of structural non-zeros in each output column.
    _cy_sparse_dot_indptr(
        A_data, A_indices, A_indptr, A_nrows,
        B_data, B_indices, B_indptr, B_ncols,
        C_indptr, zero_value,
    )

    # Convert the column counts into a valid CSC pointer array.
    C_indptr = np.cumsum(C_indptr)

    # Extract the total number of non-zero elements in the product.
    nnz = C_indptr[B_ncols]

    # Return immediately if the structural product is empty.
    if nnz == 0:
        return csc_array((A_nrows, B_ncols))

    # Define the largest signed 32-bit integer for overflow detection.
    max_32 = 2 ** 31 - 1

    # Promote structure arrays to 64-bit integers if the product is too large.
    if nnz > max_32 and dtype_I == np.int32:
        dtype_I = np.int64
        A_indices = A_indices.astype(dtype_I, copy=False)
        A_indptr = A_indptr.astype(dtype_I, copy=False)
        B_indices = B_indices.astype(dtype_I, copy=False)
        B_indptr = B_indptr.astype(dtype_I, copy=False)
        C_indptr = C_indptr.astype(dtype_I, copy=False)

    # Otherwise ensure that the output pointer array matches the index type.
    else:
        C_indptr = C_indptr.astype(dtype_I, copy=False)

    # Allocate the numerical data and row-index arrays of the result.
    C_data = np.zeros(nnz, dtype=dtype_T)
    C_indices = np.zeros(nnz, dtype=dtype_I)

    # Evaluate the sparse product into the preallocated CSC buffers.
    _cy_sparse_dot(
        A_data, A_indices, A_indptr, A_nrows,
        B_data, B_indices, B_indptr, B_ncols,
        C_data, C_indices, C_indptr,
        zero_value,
    )

    # Construct and return the SciPy CSC representation of the product.
    return csc_array(
        (C_data, C_indices, C_indptr),
        shape=(A_nrows, B_ncols),
    )