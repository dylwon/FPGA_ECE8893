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
    data_t C[N_ROWS][N_COLS];
    // Array Partitioning: 'A' and 'C' are partitioned on Dimension 2 (Columns) to enable parallel access during row-wise loop operations for Phase 1.
    #pragma HLS ARRAY_PARTITION variable=A dim=2 type=complete
    #pragma HLS ARRAY_PARTITION variable=C dim=2 type=complete

    for (int i = 0; i < N_ROWS; i++) {
        for (int j = 0; j < N_COLS; j++) {
            #pragma HLS PIPELINE II=1 // Loop Pipelining: Set Initiation Interval to 1 to start new iteration every clock cycle
            #pragma HLS LOOP_FLATTEN // Loop Flattening: Flatten nested loops to improve pipeline efficiency
            // #pragma HLS UNROLL factor=64 // Loop Unrolling: Unroll loop to process elements in parallel

            A[i][j] = A_DRAM[i][j];

            // We may be able to read faster if we increase AXI bus width
        }
    }



    // Intermediate buffer for row-normalized values
    data_t tmp[N_ROWS][N_COLS];
    #pragma HLS ARRAY_PARTITION variable=tmp dim=2 type=complete // Array Partitioning: 'tmp' is partitioned on both dimensions to enable parallel access during both row and column wise operatios for Phases 1 and 2.

    // Phase 1: Row-wise normalization
    for (int i = 0; i < N_ROWS; i++) {
        data_t row_sum = 0.0;

        // Compute row sum
        for (int j = 0; j < N_COLS; j++) {
            #pragma HLS PIPELINE II=1 // Loop Pipelining: Set Initiation Interval to 1 to start new iteration every clock cycle
            #pragma HLS UNROLL factor=64 // Loop Unrolling: Unroll loop to process elements in parallel

            row_sum += A[i][j];
        }

        // Avoid division by zero, add small bias
        data_t denom = row_sum + (data_t)1.0;

        // Normalize each element in the row
        for (int j = 0; j < N_COLS; j++) {
            #pragma HLS PIPELINE II=1 // Loop Pipelining: Set Initiation Interval to 1 to start new iteration every clock cycle
            #pragma HLS UNROLL factor=64 // Loop Unrolling: Unroll loop to process elements in parallel

            tmp[i][j] = A[i][j] / denom;
        }
    }
    


    // Phase 2: Column-wise scaling

    // Buffer to hold running sums for all columns
    data_t col_sums[N_COLS];

    // Array Partitioning: 'col_sums' is partitioned to enable parallel access during loop operations.
    #pragma HLS ARRAY_PARTITION variable=col_sums type=complete

    // Initialize sums
    for(int j=0; j<N_COLS; j++) {
        #pragma HLS PIPELINE II=1 // Loop Pipelining: Set Initiation Interval to 1 to start new iteration every clock cycle
        #pragma HLS UNROLL factor=64 // Loop Unrolling: Unroll loop to process elements in parallel

        col_sums[j] = 0;
    }

    // Loop Reordering: Perform row-column traversal for improved memory access locality
    for (int i = 0; i < N_ROWS; i++) {
        for (int j = 0; j < N_COLS; j++) {
            #pragma HLS PIPELINE II=1 // Loop Pipelining: Set Initiation Interval to 1 to start new iteration every clock cycle
            #pragma HLS UNROLL factor=64 // Loop Unrolling: Unroll loop to process elements in parallel

            col_sums[j] += tmp[i][j]; 
        }
    }

    // Final Scale Calculation and Write
    for (int i = 0; i < N_ROWS; i++) {
        for (int j = 0; j < N_COLS; j++) {
            #pragma HLS PIPELINE II=1 // Loop Pipelining: Set Initiation Interval to 1 to start new iteration every clock cycle
            #pragma HLS UNROLL factor=64 // Loop Unrolling: Unroll loop to process elements in parallel

            data_t scale = col_sums[j] / (data_t)N_ROWS;
            C[i][j] = tmp[i][j] * scale;
        }
    }

    // Write back results to DRAM
    for (int i = 0; i < N_ROWS; i++) {
        for (int j = 0; j < N_COLS; j++) {
            #pragma HLS PIPELINE II=1 // Loop Pipelining: Set Initiation Interval to 1 to start new iteration every clock cycle

            C_DRAM[i][j] = C[i][j];

            // We may be able to write back faster if we increase AXI bus width
        }
    }

}
