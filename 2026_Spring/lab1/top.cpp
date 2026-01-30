#include "dcl.h"

// HLS top-level function
void top_kernel(data_t A_DRAM[N_ROWS][N_COLS],
                data_t C_DRAM[N_ROWS][N_COLS]) {

    // Moving DRAM interfaces to BRAM
    #pragma HLS interface m_axi port=A_DRAM offset=slave bundle=A max_read_burst_length=256 num_read_outstanding=16
    #pragma HLS interface m_axi port=C_DRAM offset=slave bundle=C max_write_burst_length=256 num_write_outstanding=16
    #pragma HLS interface s_axilite port=return

    // On-chip buffers for A_DRAM and C_DRAM
    data_t A[N_ROWS][N_COLS];
    #pragma HLS ARRAY_PARTITION variable=A dim=2 type=complete

    data_t C[N_ROWS][N_COLS];
    #pragma HLS ARRAY_PARTITION variable=C dim=2 type=complete

    for (int i = 0; i < N_ROWS; i++) {
        #pragma HLS PIPELINE II=1 // Loop Pipelining: Set Initiation Interval to 1 to start new iteration every clock cycle

        for (int j = 0; j < N_COLS; j++) {
            #pragma HLS UNROLL factor=64 // Loop Unrolling: Unroll loop to process elements in parallel

            A[i][j] = A_DRAM[i][j];
        }
    }

    // Intermediate buffer for row-normalized values
    data_t tmp[N_ROWS][N_COLS];
    #pragma HLS ARRAY_PARTITION variable=tmp dim=2 type=complete // Array Partitioning: 'tmp' is partitioned on both dimensions to enable parallel access during both row and column wise operatios for Phases 1 and 2.

    data_t col_sums[N_COLS]; // Buffer to hold running sums for all columns
    #pragma HLS ARRAY_PARTITION variable=col_sums type=complete // Array Partitioning: 'col_sums' is partitioned to enable parallel access during loop operations.

    // Initialize sums
    for(int j=0; j<N_COLS; j++) {
        #pragma HLS UNROLL factor=64 // Loop Unrolling: Unroll loop to process elements in parallel

        col_sums[j] = 0;
    }

    data_t denoms[N_ROWS]; // Buffer to hold calculated denominators for all columns
    #pragma HLS ARRAY_PARTITION variable=denoms type=complete // Array Partitioning: 'denoms' is partitioned to enable parallel access during loop operations.

    // Phase 1: Row-wise normalization
    for (int i = 0; i < N_ROWS; i++) {
        #pragma HLS PIPELINE II=1 // Loop Pipelining: Set Initiation Interval to 1 to start new iteration every clock cycle

        data_t row_sum = 0.0;

        // Compute row sum
        for (int j = 0; j < N_COLS; j++) {
            #pragma HLS UNROLL factor=64 // Loop Unrolling: Unroll loop to process elements in parallel

            row_sum += A[i][j];
        }

        // Avoid division by zero, add small bias
        denoms[i] = row_sum + (data_t)1.0;
    }

    // Normalize each element in the row
    for (int i = 0; i < N_ROWS; i++) {
        for (int j = 0; j < N_COLS; j++) {
            #pragma HLS PIPELINE II=1 // Loop Pipelining Inner Loop: Partial Unrolling to allow overlapping of operations
            #pragma HLS UNROLL factor=32 // Loop Unrolling: Unroll loop to process elements in parallel

            tmp[i][j] = A[i][j] / denoms[i];
            col_sums[j] += tmp[i][j];
        }
    }

    // Phase 2: Column-wise scaling

    data_t scales[N_COLS];
    #pragma HLS ARRAY_PARTITION variable=scales type=cyclic factor=32

    for(int j=0; j<N_COLS; j++){
        #pragma HLS PIPELINE II=1 // Loop Pipelining Inner Loop: Partial Unrolling to allow overlapping of operations
        #pragma HLS UNROLL factor=32

        scales[j] = col_sums[j] / (data_t)N_ROWS;
    }

    // Final Scale Calculation and Write
    for (int i = 0; i < N_ROWS; i++) {
        #pragma HLS PIPELINE II=1 // Loop Pipelining: Set Initiation Interval to 1 to start new iteration every clock cycle
        
        for (int j = 0; j < N_COLS; j++) {
            #pragma HLS UNROLL factor=64 // Loop Unrolling: Unroll loop to process elements in parallel

            C[i][j] = tmp[i][j] * scales[j];
        }
    }

    // Write back results to DRAM
    for (int i = 0; i < N_ROWS; i++) {
        #pragma HLS PIPELINE II=1 // Loop Pipelining: Set Initiation Interval to 1 to start new iteration every clock cycle

        for (int j = 0; j < N_COLS; j++) {
            #pragma HLS UNROLL factor=64 // Loop Unrolling: Unroll loop to process elements in parallel

            C_DRAM[i][j] = C[i][j];
        }
    }

}
