#include "dcl.h"

// Baseline implementation for HLS.
// Students will optimize this (loops, memory access, etc.).
void top_kernel(data_t A[N_ROWS][N_COLS],
                data_t C[N_ROWS][N_COLS]) {

    // Array Partitioning: 'A' and 'C' are partitioned on Dimension 2 (Columns) 
    // to enable parallel access during row-wise loop operations for Phase 1.
    #pragma HLS ARRAY_PARTITION variable=A dim=2 type=cyclic factor=4
    #pragma HLS ARRAY_PARTITION variable=C dim=2 type=cyclic factor=4

    // Intermediate buffer for row-normalized values
    static data_t tmp[N_ROWS][N_COLS];

    // Array Partitioning: 'tmp' is partitioned on both dimensions to enable
    // parallel access during both row and column wise operatios for Phases 1 and 2.
    // #pragma HLS ARRAY_PARTITION variable=tmp dim=1 type=cyclic factor=4
    #pragma HLS ARRAY_PARTITION variable=tmp dim=2 type=cyclic factor=4

    // Phase 1: Row-wise normalization
    for (int i = 0; i < N_ROWS; i++) {
        data_t row_sum = 0.0;

        // Compute row sum
        for (int j = 0; j < N_COLS; j++) {
            // Loop Pipelining: Set Initiation Interval to 1 to start new iteration every clock cycle
            #pragma HLS PIPELINE II=1

            // Loop Unrolling: Unroll loop to process elements in parallel
            #pragma HLS UNROLL factor=4

            row_sum += A[i][j];
        }

        // Avoid division by zero, add small bias
        data_t denom = row_sum + (data_t)1.0;

        // Normalize each element in the row
        for (int j = 0; j < N_COLS; j++) {
            // Loop Pipelining: Set Initiation Interval to 1 to start new iteration every clock cycle
            #pragma HLS PIPELINE II=1

            // Loop Unrolling: Unroll loop to process elements in parallel
            #pragma HLS UNROLL factor=4

            tmp[i][j] = A[i][j] / denom;
        }
    }

    // // Phase 2: Column-wise scaling
    // for (int j = 0; j < N_COLS; j++) {
    //     data_t col_sum = 0.0;

    //     // Compute column sum of normalized values
    //     for (int i = 0; i < N_ROWS; i++) {
    //         // Loop Pipelining: Set Initiation Interval to 1 to start new iteration every clock cycle
    //         #pragma HLS PIPELINE II=1

    //         // Loop Unrolling: Unroll loop to process elements in parallel
    //         #pragma HLS UNROLL factor=4

    //         col_sum += tmp[i][j];
    //     }

    //     // Compute average as scale
    //     data_t scale = col_sum / (data_t)N_ROWS;

    //     // Apply scale to each element in the column
    //     for (int i = 0; i < N_ROWS; i++) {
    //         // Loop Pipelining: Set Initiation Interval to 1 to start new iteration every clock cycle
    //         #pragma HLS PIPELINE II=1

    //         // Loop Unrolling: Unroll loop to process elements in parallel
    //         #pragma HLS UNROLL factor=4

    //         C[i][j] = tmp[i][j] * scale;
    //     }
    // }
    
    // Buffer to hold running sums for all columns
    data_t col_sums[N_COLS];

    // Array Partitioning: 'col_sums' is partitioned on Dimension 2 (Columns) 
    // to enable parallel access during row-wise loop operations.
    #pragma HLS ARRAY_PARTITION variable=col_sums type=cyclic factor=4

    // Initialize sums
    for(int j=0; j<N_COLS; j++) {
        #pragma HLS PIPELINE
        #pragma HLS UNROLL factor=4
        col_sums[j] = 0;
    }

    // Loop Reordering: Perform row-column traversal for improved memory access locality
    for (int i = 0; i < N_ROWS; i++) {
        for (int j = 0; j < N_COLS; j++) {
            // Loop Pipelining: Set Initiation Interval to 1 to start new iteration every clock cycle
            #pragma HLS PIPELINE II=1

            // Loop Unrolling: Unroll loop to process elements in parallel
            #pragma HLS UNROLL factor=4

            col_sums[j] += tmp[i][j]; 
        }
    }

    // Final Scale Calculation and Write
    // Loop Reordering: Perform row-column traversal for improved memory access locality
    for (int i = 0; i < N_ROWS; i++) {
        for (int j = 0; j < N_COLS; j++) {
            // Loop Pipelining: Set Initiation Interval to 1 to start new iteration every clock cycle
            #pragma HLS PIPELINE II=1

            // Loop Unrolling: Unroll loop to process elements in parallel
            #pragma HLS UNROLL factor=4

            data_t scale = col_sums[j] / (data_t)N_ROWS;
            C[i][j] = tmp[i][j] * scale;
        }
    }
}
