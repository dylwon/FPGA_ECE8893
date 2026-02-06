#include "dcl.h"
#include <ap_int.h>

// 1. Define Vector Type (16 elements * 32-bit alignment = 512 bits)
// Note: Even though data_t is 24 bits, HLS aligns it to 32 bits in memory.
typedef ap_uint<512> uint512_dt;

void top_kernel(data_t A_DRAM[N_ROWS][N_COLS],
                data_t C_DRAM[N_ROWS][N_COLS]) {

    // -------------------------------------------------------------
    // INTERFACE: Keep Arguments Unchanged, use Internal Casting
    // -------------------------------------------------------------
    #pragma HLS interface m_axi port=A_DRAM offset=slave bundle=A max_read_burst_length=32 num_read_outstanding=16
    #pragma HLS interface m_axi port=C_DRAM offset=slave bundle=C max_write_burst_length=32 num_write_outstanding=16
    #pragma HLS interface s_axilite port=return

    // Internal Shadow Pointers for 512-bit access
    uint512_dt* A_wide = (uint512_dt*)A_DRAM;
    uint512_dt* C_wide = (uint512_dt*)C_DRAM;

    // -------------------------------------------------------------
    // LOCAL BUFFERS (Partitioned for Vector Access)
    // -------------------------------------------------------------
    data_t A[N_ROWS][N_COLS];
    #pragma HLS ARRAY_PARTITION variable=A dim=2 type=cyclic factor=16

    data_t C[N_ROWS][N_COLS];
    #pragma HLS ARRAY_PARTITION variable=C dim=2 type=cyclic factor=16

    data_t tmp[N_ROWS][N_COLS];
    #pragma HLS ARRAY_PARTITION variable=tmp dim=2 type=cyclic factor=16

    data_t denoms[N_ROWS];
    // No partition needed, accessed sequentially

    data_t col_sums[N_COLS];
    #pragma HLS ARRAY_PARTITION variable=col_sums type=cyclic factor=16

    data_t scales[N_COLS];
    #pragma HLS ARRAY_PARTITION variable=scales type=cyclic factor=16

    // -------------------------------------------------------------
    // STAGE 1: Vectorized Read (Bit-Exact Copy)
    // -------------------------------------------------------------
    READ_LOOP: for (int i = 0; i < N_ROWS; i++) {
        for (int j = 0; j < N_COLS / 16; j++) {
            #pragma HLS PIPELINE II=1
            
            // Read 512 raw bits
            uint512_dt raw = A_wide[i*(N_COLS/16) + j];
            
            // Unpack exactly 16 values
            for (int k = 0; k < 16; k++) {
                #pragma HLS UNROLL
                
                // CRITICAL FIX: Use .range() to copy BITS, not value.
                // This prevents HLS from trying to convert integer '5' to fixed '5.0'
                // We assume data_t is aligned to 32-bit boundaries in DRAM.
                ap_int<32> raw_bits = raw.range(31 + k*32, k*32);
                
                // Copy the bottom 24 bits (width of data_t) directly
                A[i][j*16 + k].range(23, 0) = raw_bits.range(23, 0);
            }
        }
    }

    // -------------------------------------------------------------
    // STAGE 2: Row Sums & Denom Calculation
    // -------------------------------------------------------------
    ROW_PROCESS: for (int i = 0; i < N_ROWS; i++) {
        #pragma HLS PIPELINE II=1
        
        data_t row_sum = 0;
        for (int j = 0; j < N_COLS; j++) {
            // Keep original unroll factor to match original adder tree structure
            #pragma HLS UNROLL factor=64 
            row_sum += A[i][j];
        }
        
        // CRITICAL FIX: REVERT TO ADDITION
        // Do NOT calculate reciprocal (1.0/x) here. 
        // We must store the exact denominator to use in Division later.
        denoms[i] = row_sum + (data_t)1.0;
    }

    // Initialize Accumulators
    for(int j=0; j<N_COLS; j++) {
        #pragma HLS UNROLL factor=16
        col_sums[j] = 0;
    }

    // -------------------------------------------------------------
    // STAGE 3: Tiled Normalization (Using DIVISION)
    // -------------------------------------------------------------
    COL_PROCESS: for (int i = 0; i < N_ROWS; i++) {
        data_t denom = denoms[i];

        for (int j = 0; j < N_COLS; j+=16) {
            #pragma HLS PIPELINE II=1
            
            for (int k = 0; k < 16; k++) {
                tmp[i][j+k] = A[i][j+k] / denom;
            }

            for (int k = 0; k < 16; k++) {
                col_sums[j+k] += val;
            }
        }
    }

    // -------------------------------------------------------------
    // STAGE 4: Compute Scales (Using DIVISION)
    // -------------------------------------------------------------
    COMPUTE_SCALES: for(int j=0; j<N_COLS; j++){
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL factor=16
        // Bit-exact division matches original code
        scales[j] = col_sums[j] / (data_t)N_ROWS; 
    }

    // -------------------------------------------------------------
    // STAGE 5: Vectorized Write Back (Bit-Exact Copy)
    // -------------------------------------------------------------
    WRITE_LOOP: for (int i = 0; i < N_ROWS; i++) {
        for (int j = 0; j < N_COLS / 16; j++) {
            #pragma HLS PIPELINE II=1
            
            uint512_dt raw_out;
            
            for (int k = 0; k < 16; k++) {
                #pragma HLS UNROLL
                
                // Perform math
                data_t res = tmp[i][j*16 + k] * scales[j*16 + k];
                
                // Pack bits directly using .range()
                // Do not cast to float or int, just copy the container bits
                ap_int<32> out_bits = 0;
                out_bits.range(23, 0) = res.range(23, 0);
                
                raw_out.range(31 + k*32, k*32) = out_bits;
            }
            
            C_wide[i*(N_COLS/16) + j] = raw_out;
        }
    }
}
