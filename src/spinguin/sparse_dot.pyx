# Imports
import numpy as np
cimport cython
from openmp cimport omp_get_max_threads, omp_get_thread_num
from cython.parallel import prange

# Import the absolute value function
cdef extern from "<complex>" nogil:
    double abs(double complex)

@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef sparse_dot(const double complex[::1] A_data, const long long[::1] A_indices, const long long[::1] A_indptr, const long long A_nrows,
                 const double complex[::1] B_data, const long long[::1] B_indices, const long long[::1] B_indptr, const long long B_ncols,
                 const double zero_value):
    """
    Memory-saving sparse matrix multiplication C = A*B algorithm compatible with SciPy CSC arrays.

    Parameters
    ----------
    A_data : numpy.ndarray
        The data array for matrix A obtained from SciPy by A.data. Data type must be np.complex128.
    A_indices : numpy.ndarray
        The index array for matrix A obtained from SciPy by A.indices. Data type must be np.longlong.
    A_indptr : numpy.ndarray
        The index pointer array for matrix A obtained from SciPy by A.indptr. Data type must be np.longlong.
    A_nrows : int
        The number of rows in the matrix A. Data type must be np.longlong.
    B_data : numpy.ndarray
        The data array for matrix B obtained from SciPy by B.data. Data type must be np.complex128.
    B_indices : numpy.ndarray
        The index array for matrix B obtained from SciPy by B.indices. Data type must be np.longlong.
    B_indptr : numpy.ndarray
        The index pointer array for matrix B obtained from SciPy by B.indptr. Data type must be np.longlong.
    B_ncols : int
        The number of columns in the matrix B. Data type must be np.longlong.
    zero_value : float
        Matrix elements less than zero_value will be considered as zero in the returned matrix C.
        Data type must be np.float64.

    Returns
    -------
    C_data : numpy.ndarray
        The data array of result matrix.
    C_indices : numpy.ndarray
        The index array of result matrix.
    C_indptr : numpy.ndarray
        The index pointer array of result matrix.
    """

    # Obtain the number of threads to use
    cdef long long num_threads
    cdef long long thread_id
    num_threads = omp_get_max_threads()

    # Allocate memory for result index pointers
    cdef long long[::1] C_indptr = np.zeros(B_ncols+1, dtype=np.longlong)
    
    # Allocate memory for storing the results of current column
    cdef double complex[::1] C_col_data = np.zeros((num_threads*A_nrows), dtype=np.complex128)   # Used to store the data
    cdef long long[::1] C_col_nonzero = np.zeros((num_threads*A_nrows), dtype=np.longlong)       # Used to store, whether a value COULD be non-zero 0 = zero, 1 = non-zero
    cdef long long[::1] C_col_indices = np.zeros((num_threads*A_nrows), dtype=np.longlong)       # Used to store, where the non-zero values are located

    # Initialize loop variables and counters
    cdef long long nnz
    cdef long long C_col_nnz
    cdef long long nnz_th
    cdef long long i
    cdef long long j
    cdef long long k
    cdef long long start_A
    cdef long long start_B
    cdef long long end_A
    cdef long long end_B
    cdef long long ind_j
    cdef long long ind_k
    cdef long long ind_k_thread
    cdef long long thread_start
    cdef double complex val_j
    cdef double complex val_k
        
    # FIRST LOOP - TO FIND THE NUMBER OF NON-ZERO ELEMENTS AND TO CONSTRUCT THE INDEX POINTER ARRAY

    # Go through the columns of matrix B
    for i in prange(B_ncols, nogil=True, schedule='static'):

        # Get the thread id
        thread_id = omp_get_thread_num()

        # Calculate the thread starting index
        thread_start = thread_id * A_nrows

        # Counter for the non-zero values in current column (thread-local)
        C_col_nnz = 0

        # Obtain the starting and ending indices of the current column of B
        start_B = B_indptr[i]
        end_B = B_indptr[i+1]

        # Loop through the column of B
        for j in range(start_B, end_B):

            # Get the row index and the value
            ind_j = B_indices[j]
            val_j = B_data[j]

            # Find the column from A, to which the current element is multiplying
            start_A = A_indptr[ind_j]
            end_A = A_indptr[ind_j+1]

            # Loop through the column of A
            for k in range(start_A, end_A):

                # Get the row index and the value
                ind_k = A_indices[k]
                ind_k_thread = ind_k + thread_start
                val_k = A_data[k]

                # Multiply, and add to the array
                C_col_data[ind_k_thread] = C_col_data[ind_k_thread] + val_j * val_k
                
                # Check if the non-zero value is found for the matrix element for the first time
                if C_col_nonzero[ind_k_thread] == 0:
                    C_col_indices[C_col_nnz + thread_start] = ind_k
                    C_col_nonzero[ind_k_thread] = 1
                    C_col_nnz = C_col_nnz + 1

        # Counter for number of non-zeros greater than the threshold (thread-local)
        nnz_th = 0

        # Once a complete column is calculated, go through the possible non-zero values
        for k in range(C_col_nnz):

            # Get the row index and the value
            ind_k = C_col_indices[k + thread_start]
            ind_k_thread = ind_k + thread_start
            val_k = C_col_data[ind_k_thread]

            # Increase the counters if value is larger than threshold
            if abs(val_k) > zero_value:
                nnz_th = nnz_th + 1

            # Clear the arrays
            C_col_data[ind_k_thread] = 0
            C_col_nonzero[ind_k_thread] = 0

        # Append the number of non-zeros of the column to the index pointer array
        C_indptr[i+1] = nnz_th

    # Calculate the cumulative sum (the true index pointers)
    for i in range(B_ncols):
        C_indptr[i+1] = C_indptr[i+1] + C_indptr[i]

    # Get the number of non-zeros
    nnz = C_indptr[B_ncols]

    # SECOND LOOP - TO ASSIGN THE ELEMENTS

    # Allocate memory for result data and indices
    cdef double complex [::1] C_data = np.zeros(nnz, dtype=np.complex128)
    cdef long long [::1] C_indices = np.zeros(nnz, dtype=np.longlong)

    # Go through the columns of matrix B
    for i in prange(B_ncols, nogil=True, schedule='static'):

        # Get the thread id
        thread_id = omp_get_thread_num()

        # Calculate the thread starting index
        thread_start = thread_id * A_nrows

        # Counter for the non-zero values in current column (thread-local)
        C_col_nnz = 0

        # Obtain the starting and ending indices of the current column of B
        start_B = B_indptr[i]
        end_B = B_indptr[i+1]

        # Loop through the column of B
        for j in range(start_B, end_B):

            # Get the row index and the value
            ind_j = B_indices[j]
            val_j = B_data[j]

            # Find the column from A, to which the current element is multiplying
            start_A = A_indptr[ind_j]
            end_A = A_indptr[ind_j+1]

            # Loop through the column of A
            for k in range(start_A, end_A):

                # Get the row index and the value
                ind_k = A_indices[k]
                ind_k_thread = ind_k + thread_start
                val_k = A_data[k]

                # Multiply, and add to the array
                C_col_data[ind_k_thread] = C_col_data[ind_k_thread] + val_j * val_k
                
                # Check if the non-zero value is found for the matrix element for the first time
                if C_col_nonzero[ind_k_thread] == 0:
                    C_col_indices[C_col_nnz + thread_start] = ind_k
                    C_col_nonzero[ind_k_thread] = 1
                    C_col_nnz = C_col_nnz + 1

        # Counter for number of non-zeros greater than the threshold (thread-local)
        nnz_th = C_indptr[i]
        
        # Once a complete column is calculated, go through the possible non-zero values
        for k in range(C_col_nnz):

            # Get the row index and the value
            ind_k = C_col_indices[k + thread_start]
            ind_k_thread = ind_k + thread_start
            val_k = C_col_data[ind_k_thread]

            # Add to the array if value is larger than threshold
            if abs(val_k) > zero_value:
                C_data[nnz_th] = val_k
                C_indices[nnz_th] = ind_k
                nnz_th = nnz_th + 1

            # Clear the arrays
            C_col_data[ind_k_thread] = 0
            C_col_nonzero[ind_k_thread] = 0

    # Convert the memoryviews back to arrays
    C_data = np.asarray(C_data)
    C_indices = np.asarray(C_indices)
    C_indptr = np.asarray(C_indptr)

    return C_data, C_indices, C_indptr