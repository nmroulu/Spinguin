"""
This module contains a Cython function for finding the intersection of two 
sorted 2D arrays.
"""

# Imports
import numpy as np
cimport cython

@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def intersect_indices(const long long[::1] A,
                      const long long[::1] B,
                      const long long row_len) -> tuple[np.ndarray, np.ndarray]:
    """
    Fast O(n) implementation for finding the indices of common rows from two 2D
    arrays. The arrays must be pre-prepared into contiguous 1D format. Each row
    in the original 2D array must be unique and they must be sorted in
    lexicographic order.

    Parameters
    ----------
    A : ndarray
        First array converted to contiguous 1D format. Data type must be
        np.longlong.
    B : ndarray
        Second array converted to contiguous 1D format. Data type must be
        np.longlong.
    row_len : int
        Length of the rows. Data type must be np.longlong.

    Returns
    -------
    A_ind : ndarray
        Indices of the common elements in array `A`.
    B_ind : ndarray
        Indices of the common elements in array `B`.
    """

    # Obtain the lengths of the arrays
    cdef long long A_len = A.shape[0]
    cdef long long B_len = B.shape[0]

    # Initialize pointers for A, B, and the result arrays
    cdef long long A_ptr = 0
    cdef long long B_ptr = 0
    cdef long long common_ptr = 0

    # Initialize the column pointers
    cdef long long col = 0

    # Initialize boolean for common element
    cdef long long common_element

    # Allocate memory for the result indices
    cdef long long result_len
    if A_len < B_len:
        result_len = A_len // row_len
    else:
        result_len = B_len // row_len
    cdef long long [::1] A_ind = np.zeros(result_len, dtype=np.longlong)
    cdef long long [::1] B_ind = np.zeros(result_len, dtype=np.longlong)

    # Loop until we reach the end of either array
    while True:

        # Set common element to True by default
        common_element = 1

        # Check whether we have a common element
        for col in range(row_len):

            # If A is larger, move B pointer to next row, set common element to
            # False and exit loop
            if A[col + A_ptr*row_len] > B[col + B_ptr*row_len]:
                B_ptr = B_ptr + 1
                common_element = 0
                break

            # If B is larger, move A pointer to next row, set common element to
            # False and exit loop
            elif B[col + B_ptr*row_len] > A[col + A_ptr*row_len]:
                A_ptr = A_ptr + 1
                common_element = 0
                break

        # If common element is found, save indices and move all pointers
        if common_element == 1:
            A_ind[common_ptr] = A_ptr
            B_ind[common_ptr] = B_ptr
            A_ptr = A_ptr + 1
            B_ptr = B_ptr + 1
            common_ptr = common_ptr + 1

        # Check whether we have arrived to the end of comparison
        if (A_ptr*row_len == A_len) or (B_ptr*row_len == B_len):
            break

    # Return the results
    return np.asarray(A_ind[0:common_ptr]), np.asarray(B_ind[0:common_ptr])