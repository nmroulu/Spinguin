#ifndef CSPDOT_HPP
#define CSPDOT_HPP

#include <vector>
#include <complex>
#include <omp.h>

// Real numbers
template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, T>::type
my_abs(T x) {
    return x < T(0) ? -x : x;
}

// Complex numbers
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
    #pragma omp parallel
    {
        // Thread-local memory for results of current column
        std::vector<T> C_col_data(A_nrows, 0);
        std::vector<I> C_col_nonzero(A_nrows, -1);

        // Iterate through the columns of matrix B
        #pragma omp for
        for (I i = 0; i < B_ncols; i++){

            // Current head of the column
            I C_col_head = -2;

            // Obtain the starting and ending indices of the current column of B
            I start_B = B_indptr[i];
            I end_B = B_indptr[i+1];

            // Loop through the column of B
            for (I j = start_B; j < end_B; j++){

                // Get the row index and value
                I ind_j = B_indices[j];
                T val_j = B_data[j];

                // Find the column from A that the current element is multiplying
                I start_A = A_indptr[ind_j];
                I end_A = A_indptr[ind_j + 1];

                // Loop through the column of A
                for (I k = start_A; k < end_A; k++){

                    // Get the row index and the value
                    I ind_k = A_indices[k];
                    T val_k = A_data[k];

                    // Multiply and add to the array
                    C_col_data[ind_k] = C_col_data[ind_k] + val_j * val_k;

                    // Check if a non-zero value is found for the matrix element for the first time
                    if (C_col_nonzero[ind_k] == -1){
                        C_col_nonzero[ind_k] = C_col_head;
                        C_col_head = ind_k;
                    }
                }
            }

            // Counter for the number of non-zeros
            I nnz = 0;

            // Once a complete column is calculated, iterate through the possible non-zero values
            while (C_col_head != -2) {

                // Get the value
                T val_k = C_col_data[C_col_head];

                // Increment the counter if the value is larger than the threshold
                if (my_abs(val_k) > zero_value){
                    nnz++;
                }

                // Get the next index
                I C_col_head_temp = C_col_head;
                C_col_head = C_col_nonzero[C_col_head];

                // Clear the arrays
                C_col_data[C_col_head_temp] = 0;
                C_col_nonzero[C_col_head_temp] = -1;
            }

            // Append the number of non-zeros to the index pointer array
            C_indptr[i+1] = nnz;
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
    #pragma omp parallel
    {
        // Thread-local memory for results of current column
        std::vector<T> C_col_data(A_nrows, 0);
        std::vector<I> C_col_nonzero(A_nrows, -1);

        // Iterate through the columns of matrix B
        #pragma omp for
        for (I i = 0; i < B_ncols; i++){

            // Current head of the column
            I C_col_head = -2;

            // Obtain the starting and ending indices of the current column of B
            I start_B = B_indptr[i];
            I end_B = B_indptr[i+1];

            // Loop through the column of B
            for (I j = start_B; j < end_B; j++){

                // Get the row index and value
                I ind_j = B_indices[j];
                T val_j = B_data[j];

                // Find the column from A that the current element is multiplying
                I start_A = A_indptr[ind_j];
                I end_A = A_indptr[ind_j + 1];

                // Loop through the column of A
                for (I k = start_A; k < end_A; k++){

                    // Get the row index and the value
                    I ind_k = A_indices[k];
                    T val_k = A_data[k];

                    // Multiply and add to the array
                    C_col_data[ind_k] = C_col_data[ind_k] + val_j * val_k;

                    // Check if a non-zero value is found for the matrix element for the first time
                    if (C_col_nonzero[ind_k] == -1){
                        C_col_nonzero[ind_k] = C_col_head;
                        C_col_head = ind_k;
                    }
                }
            }

            // Counter for the number of non-zeros
            I nnz = C_indptr[i];

            // Once a complete column is calculated, iterate through the possible non-zero values
            while (C_col_head != -2) {

                // Get the value
                T val_k = C_col_data[C_col_head];

                // Add to the array if the value is larger than the threshold
                if (my_abs(val_k) > zero_value){
                    C_data[nnz] = val_k;
                    C_indices[nnz] = C_col_head;
                    nnz++;
                }

                // Get the next index
                I C_col_head_temp = C_col_head;
                C_col_head = C_col_nonzero[C_col_head];

                // Clear the arrays
                C_col_data[C_col_head_temp] = 0;
                C_col_nonzero[C_col_head_temp] = -1;
            }
        }
    }
}

#endif