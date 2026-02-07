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
    #pragma HLS ARRAY_PARTITION variable=A dim=2 type=cyclic factor=16

    data_t C[N_ROWS][N_COLS];
    #pragma HLS ARRAY_PARTITION variable=C dim=2 type=cyclic factor=16

    A_BRAM_WRITE: for (int i = 0; i < N_ROWS; i++) {
        #pragma HLS PIPELINE II=1

        for (int j = 0; j < N_COLS; j++) {
            A[i][j] = A_DRAM[i][j];
        }
    }

    // Intermediate buffer for row-normalized values
    data_t tmp[N_ROWS][N_COLS];
    #pragma HLS ARRAY_PARTITION variable=tmp dim=2 type=cyclic factor=16

    // Buffer to hold calculated denominators for all columns
    data_t denoms[N_ROWS]; 
    #pragma HLS ARRAY_PARTITION variable=denoms type=cyclic factor=16

    // Phase 1: Row-wise normalization
    ROW_SUMS: for (int i = 0; i < N_ROWS; i++) {
        #pragma HLS PIPELINE II=1

        data_t row_sum = 0.0;

        // Compute row sum
        for (int j = 0; j < N_COLS; j++) {
            row_sum += A[i][j];
        }

        // Avoid division by zero, add small bias
        denoms[i] = row_sum + (data_t)1.0;
    }

    // Buffer to hold running sums for all columns
    data_t col_sums[N_COLS]; 
    #pragma HLS ARRAY_PARTITION variable=col_sums type=cyclic factor=16

    // Initialize sums
    INIT_COL_SUMS: for (int j = 0; j < N_COLS; j++) {
        #pragma HLS UNROLL factor=16

        col_sums[j] = 0;
    }

    // Normalize each element in the row
    COL_SUMS: for (int i = 0; i < N_ROWS; i++) {
        data_t denom = denoms[i];

        for (int j = 0; j < N_COLS; j += 16) {
            #pragma HLS PIPELINE II=1

            for (int k = 0; k < 16; k++) {
                tmp[i][j + k] = A[i][j + k] / denom;
            }

            for (int k = 0; k < 16; k++) {
                col_sums[j + k] += tmp[i][j + k];
            }
        }
    }

    // Phase 2: Column-wise scaling
    data_t scales[N_COLS];
    #pragma HLS ARRAY_PARTITION variable=scales type=cyclic factor=16

    SCALE_NORMALIZE: for (int j = 0; j < N_COLS; j += 16) {
        #pragma HLS PIPELINE II=1

        for (int k = 0; k < 16; k++) {
            scales[j + k] = col_sums[j + k] / (data_t)N_ROWS;
        }
    }

    // Write back results to DRAM
    C_DRAM_WRITE: for (int i = 0; i < N_ROWS; i++) {

        for (int j = 0; j < N_COLS; j += 16) {
            #pragma HLS PIPELINE II=1

            for (int k = 0; k < 16; k++) {
                C_DRAM[i][j + k] = tmp[i][j + k] * scales[j + k];
            }
        }
    }
}
