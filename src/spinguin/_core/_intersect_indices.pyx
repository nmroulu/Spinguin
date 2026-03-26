"""
intersect_indices.pyx

Provides a Cython helper for finding the common rows of two sorted arrays that
have been flattened into contiguous one-dimensional storage.
"""

# Imports
import numpy as np
cimport cython


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def intersect_indices(
    const long long[::1] A,
    const long long[::1] B,
    const long long row_len,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the indices of rows that are common to two sorted arrays.

    This is an $O(n)$ implementation for arrays that have already been
    converted to contiguous one-dimensional format. Each row in the original
    two-dimensional arrays must be unique, and the rows must be sorted in
    lexicographic order.

    Parameters
    ----------
    A : ndarray
        First array converted to contiguous one-dimensional format. Data type
        must be
        np.longlong.
    B : ndarray
        Second array converted to contiguous one-dimensional format. Data type
        must be
        np.longlong.
    row_len : int
        Number of elements in each original row. Data type must be
        np.longlong.

    Returns
    -------
    A_ind : ndarray
        Row indices of the common elements in array `A`.
    B_ind : ndarray
        Row indices of the common elements in array `B`.
    """

    # Obtain the flattened lengths of the two input arrays.
    cdef long long A_len = A.shape[0]
    cdef long long B_len = B.shape[0]

    # Initialise row pointers for the two inputs and the output arrays.
    cdef long long A_ptr = 0
    cdef long long B_ptr = 0
    cdef long long common_ptr = 0

    # Initialise the column pointer used for row-wise comparison.
    cdef long long col = 0

    # Track whether the current rows match completely.
    cdef long long common_element

    # Track the current row offsets in the flattened arrays.
    cdef long long A_offset = 0
    cdef long long B_offset = 0

    # Allocate enough space for the largest possible intersection.
    cdef long long result_len
    if A_len < B_len:
        result_len = A_len // row_len
    else:
        result_len = B_len // row_len
    cdef long long [::1] A_ind = np.zeros(result_len, dtype=np.longlong)
    cdef long long [::1] B_ind = np.zeros(result_len, dtype=np.longlong)

    # Advance through both arrays until one input is exhausted.
    while (A_ptr * row_len < A_len) and (B_ptr * row_len < B_len):

        # Cache the flattened offsets of the current rows.
        A_offset = A_ptr * row_len
        B_offset = B_ptr * row_len

        # Assume that the current rows match until proven otherwise.
        common_element = 1

        # Compare the current rows element by element.
        for col in range(row_len):

            # Advance in B when the current row of A is lexicographically larger.
            if A[col + A_offset] > B[col + B_offset]:
                B_ptr = B_ptr + 1
                common_element = 0
                break

            # Advance in A when the current row of B is lexicographically larger.
            elif B[col + B_offset] > A[col + A_offset]:
                A_ptr = A_ptr + 1
                common_element = 0
                break

        # Store matching row indices and advance both row pointers.
        if common_element == 1:
            A_ind[common_ptr] = A_ptr
            B_ind[common_ptr] = B_ptr
            A_ptr = A_ptr + 1
            B_ptr = B_ptr + 1
            common_ptr = common_ptr + 1

    # Return the populated prefixes of the index arrays.
    return np.asarray(A_ind[0:common_ptr]), np.asarray(B_ind[0:common_ptr])