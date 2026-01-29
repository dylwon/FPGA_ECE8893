#include "dcl.h"

// Helper function for Burst Reads
void read_input(data_t A_DRAM[N_ROWS][N_COLS], data_t A[N_ROWS][N_COLS]) {
    for (int i = 0; i < N_ROWS; i++) {
        #pragma HLS PIPELINE II=1 // Loop Pipelining: Set Initiation Interval to 1 to start new iteration every clock cycle
        for (int j = 0; j < N_COLS; j++) {
            A[i][j] = A_DRAM[i][j];
        }
    }
}

void process_kernel(data_t A[N_ROWS][N_COLS], data_t C[N_ROWS][N_COLS]) {
    data_t tmp[N_ROWS][N_COLS];
    #pragma HLS ARRAY_PARTITION variable=tmp dim=2 type=complete // Array Partitioning: 'tmp' is partitioned to enable parallel access during loop operations.
    
    data_t col_sums[N_COLS] = {0};
    #pragma HLS ARRAY_PARTITION variable=col_sums type=complete // Array Partitioning: 'col_sums' is partitioned to enable parallel access during loop operations.

    // Phase 1: Row-wise normalization
    for (int i = 0; i < N_ROWS; i++) {
        #pragma HLS PIPELINE II=1 // Loop Pipelining: Set Initiation Interval to 1 to start new iteration every clock cycle

        data_t row_sum = 0.0;

        // Compute row sum with unrolling to allow an adder tree
        for (int j = 0; j < N_COLS; j++) {
            #pragma HLS UNROLL factor=64

            row_sum += A[i][j];
        }

        // Avoid division by zero, add small bias
        data_t denom = row_sum + (data_t)1.0;

        for (int j = 0; j < N_COLS; j++) {
            #pragma HLS PIPELINE II=1 // Loop Pipelining: Set Initiation Interval to 1 to start new iteration every clock cycle
            #pragma HLS UNROLL factor=16 // Partial unroll to balance division latency

            tmp[i][j] = A[i][j] / denom;
            col_sums[j] += tmp[i][j]; // Accumulate col_sum here to save a loop!
        }
    }

    // Phase 2: Column scaling and final result

    data_t scales[N_COLS] = {0};
    #pragma HLS ARRAY_PARTITION variable=scales type=complete // Array Partitioning: 'scales' is partitioned to enable parallel access during loop operations.

    for (int j = 0; j < N_COLS; j++) {
        #pragma HLS PIPELINE II=1 // Loop Pipelining: Set Initiation Interval to 1 to start new iteration every clock cycle
        #pragma HLS UNROLL factor=16 // Partial unroll to balance division latency

        scales[j] = col_sums[j] / (data_t)N_ROWS;
    }

    for (int i = 0; i < N_ROWS; i++) {
        #pragma HLS PIPELINE II=1 // Loop Pipelining: Set Initiation Interval to 1 to start new iteration every clock cycle

        for (int j = 0; j < N_COLS; j++) {
            #pragma HLS UNROLL factor=64

            C[i][j] = tmp[i][j] * scales[j];
        }
    }
}

void top_kernel(data_t A_DRAM[N_ROWS][N_COLS], data_t C_DRAM[N_ROWS][N_COLS]) {
    #pragma HLS INTERFACE m_axi port=A_DRAM offset=slave bundle=A
    #pragma HLS INTERFACE m_axi port=C_DRAM offset=slave bundle=C
    #pragma HLS INTERFACE s_axilite port=return

    data_t A[N_ROWS][N_COLS];
    data_t C[N_ROWS][N_COLS];
    #pragma HLS ARRAY_PARTITION variable=A dim=2 type=complete // Array Partitioning: 'A' is partitioned to enable parallel access during loop operations.
    #pragma HLS ARRAY_PARTITION variable=C dim=2 type=complete // Array Partitioning: 'C' is partitioned to enable parallel access during loop operations.

    #pragma HLS DATAFLOW

    read_input(A_DRAM, A);
    process_kernel(A, C);

    // Write back results to DRAM
    for (int i = 0; i < N_ROWS; i++) {
        for (int j = 0; j < N_COLS; j++) {
            #pragma HLS PIPELINE II=1 // Loop Pipelining: Set Initiation Interval to 1 to start new iteration every clock cycle

            C_DRAM[i][j] = C[i][j];

            // We may be able to write back faster if we increase AXI bus width
        }
    }
}
