/*
Implementation of the low-level sparse matrix multiplication kernels used
by the Cython wrapper in `_sparse_dot.pyx`. The routines operate on CSC-form
matrix data and use thread-local work buffers together with OpenMP
parallelisation to construct the product matrix column by column.
*/

#ifndef CSPDOT_HPP
#define CSPDOT_HPP

#include <cmath>
#include <complex>
#include <omp.h>
#include <type_traits>
#include <vector>

// Return the absolute value of a real number.
template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, T>::type
my_abs(T x) {
    return x < T(0) ? -x : x;
}

// Return the magnitude of a complex number.
template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type
my_abs(const std::complex<T>& z) {
    T x = z.real();
    T y = z.imag();
    return std::sqrt(x * x + y * y);
}

template <typename I, typename T>
void c_sparse_dot_indptr(
    const T* A_data,
    const I* A_indices,
    const I* A_indptr,
    const I A_nrows,
    const T* B_data,
    const I* B_indices,
    const I* B_indptr,
    const I B_ncols,
    I* C_indptr,
    const double zero_value
)
{
    // Define sentinels for the sparse linked-list representation.
    const I unseen = static_cast<I>(-1);
    const I list_end = static_cast<I>(-2);

    #pragma omp parallel
    {
        // Allocate thread-local buffers for one output column at a time.
        std::vector<T> C_col_data(A_nrows, 0);
        std::vector<I> C_col_nonzero(A_nrows, unseen);

        // Process the columns of matrix B in parallel.
        #pragma omp for
        for (I i = 0; i < B_ncols; i++) {

            // Initialise the linked list of non-zero candidates.
            I C_col_head = list_end;

            // Obtain the CSC bounds of the current column of B.
            I start_B = B_indptr[i];
            I end_B = B_indptr[i + 1];

            // Accumulate A times the current column of B into the work array.
            for (I j = start_B; j < end_B; j++) {

                // Read the current non-zero entry of B.
                I ind_j = B_indices[j];
                T val_j = B_data[j];

                // Locate the corresponding column of A.
                I start_A = A_indptr[ind_j];
                I end_A = A_indptr[ind_j + 1];

                // Scatter the product contribution into the output column.
                for (I k = start_A; k < end_A; k++) {

                    // Read the current non-zero entry of A.
                    I ind_k = A_indices[k];
                    T val_k = A_data[k];

                    // Accumulate the product contribution at the output row.
                    C_col_data[ind_k] = C_col_data[ind_k] + val_j * val_k;

                    // Insert the row into the linked list on first visit.
                    if (C_col_nonzero[ind_k] == unseen) {
                        C_col_nonzero[ind_k] = C_col_head;
                        C_col_head = ind_k;
                    }
                }
            }

            // Count how many retained non-zero entries the column contains.
            I nnz = 0;

            // Traverse the linked list of candidate rows.
            while (C_col_head != list_end) {

                // Read the accumulated value at the current output row.
                T val_k = C_col_data[C_col_head];

                // Count entries whose magnitude exceeds the threshold.
                if (my_abs(val_k) > zero_value) {
                    nnz++;
                }

                // Advance to the next linked-list node.
                I C_col_head_temp = C_col_head;
                C_col_head = C_col_nonzero[C_col_head];

                // Clear the work buffers for reuse in the next column.
                C_col_data[C_col_head_temp] = 0;
                C_col_nonzero[C_col_head_temp] = unseen;
            }

            // Store the column non-zero count in the index pointer array.
            C_indptr[i + 1] = nnz;
        }
    }
}

template <typename I, typename T>
void c_sparse_dot(
    const T* A_data,
    const I* A_indices,
    const I* A_indptr,
    const I A_nrows,
    const T* B_data,
    const I* B_indices,
    const I* B_indptr,
    const I B_ncols,
    T* C_data,
    I* C_indices,
    const I* C_indptr,
    const double zero_value
)
{
    // Define sentinels for the sparse linked-list representation.
    const I unseen = static_cast<I>(-1);
    const I list_end = static_cast<I>(-2);

    #pragma omp parallel
    {
        // Allocate thread-local buffers for one output column at a time.
        std::vector<T> C_col_data(A_nrows, 0);
        std::vector<I> C_col_nonzero(A_nrows, unseen);

        // Process the columns of matrix B in parallel.
        #pragma omp for
        for (I i = 0; i < B_ncols; i++) {

            // Initialise the linked list of non-zero candidates.
            I C_col_head = list_end;

            // Obtain the CSC bounds of the current column of B.
            I start_B = B_indptr[i];
            I end_B = B_indptr[i + 1];

            // Accumulate A times the current column of B into the work array.
            for (I j = start_B; j < end_B; j++) {

                // Read the current non-zero entry of B.
                I ind_j = B_indices[j];
                T val_j = B_data[j];

                // Locate the corresponding column of A.
                I start_A = A_indptr[ind_j];
                I end_A = A_indptr[ind_j + 1];

                // Scatter the product contribution into the output column.
                for (I k = start_A; k < end_A; k++) {

                    // Read the current non-zero entry of A.
                    I ind_k = A_indices[k];
                    T val_k = A_data[k];

                    // Accumulate the product contribution at the output row.
                    C_col_data[ind_k] = C_col_data[ind_k] + val_j * val_k;

                    // Insert the row into the linked list on first visit.
                    if (C_col_nonzero[ind_k] == unseen) {
                        C_col_nonzero[ind_k] = C_col_head;
                        C_col_head = ind_k;
                    }
                }
            }

            // Initialise the write cursor for the current output column.
            I nnz = C_indptr[i];

            // Traverse the linked list of candidate rows.
            while (C_col_head != list_end) {

                // Read the accumulated value at the current output row.
                T val_k = C_col_data[C_col_head];

                // Write entries whose magnitude exceeds the threshold.
                if (my_abs(val_k) > zero_value) {
                    C_data[nnz] = val_k;
                    C_indices[nnz] = C_col_head;
                    nnz++;
                }

                // Advance to the next linked-list node.
                I C_col_head_temp = C_col_head;
                C_col_head = C_col_nonzero[C_col_head];

                // Clear the work buffers for reuse in the next column.
                C_col_data[C_col_head_temp] = 0;
                C_col_nonzero[C_col_head_temp] = unseen;
            }
        }
    }
}

#endif