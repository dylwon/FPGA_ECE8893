#include "dcl.h"
#include <hls_stream.h>
#include <ap_int.h>

// The "Goldilocks" zone for the ZU3EG: 256 bits = 8 elements per cycle
#define TILE_FACTOR 8
#define VEC_COLS (NY / TILE_FACTOR)
#define TOTAL_BLOCKS (NX * VEC_COLS)

typedef ap_uint<256> uint256_dt;

// --------------------------------------------------------
// TASK 1: Load Data with Throttled AXI Bursts
// --------------------------------------------------------
void load_data(const data_t A_in[NX][NY], hls::stream<uint256_dt>& out_stream) {
    #pragma HLS INLINE off
    const uint256_dt* in_ptr = (const uint256_dt*)A_in;
    
    load_loop: for (int i = 0; i < TOTAL_BLOCKS; i++) {
        #pragma HLS PIPELINE II=1
        out_stream.write(in_ptr[i]);
    }
}

// --------------------------------------------------------
// TASK 2: 30-Step Compute Engine (8-wide Vectorization)
// --------------------------------------------------------
void process_data(hls::stream<uint256_dt>& in_stream, hls::stream<uint256_dt>& out_stream) {
    #pragma HLS INLINE off
    
    // Ping-pong buffer: explicitly partitioned to prevent dual-port conflicts
    uint256_dt buf_work[2][TOTAL_BLOCKS];
    #pragma HLS BIND_STORAGE variable=buf_work type=ram_2p impl=bram
    #pragma HLS ARRAY_PARTITION variable=buf_work dim=1 complete
    #pragma HLS DEPENDENCE variable=buf_work type=inter false

    // Line buffers for the 3x3 sliding window
    uint256_dt line_buf0[VEC_COLS];
    uint256_dt line_buf1[VEC_COLS];
    #pragma HLS BIND_STORAGE variable=line_buf0 type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=line_buf1 type=ram_2p impl=bram

    // 3x24 Window (Prev 8, Curr 8, Next 8) 
    data_t window[3][24];
    #pragma HLS ARRAY_PARTITION variable=window dim=0 complete

    const data_t wc = (data_t)0.50;
    const data_t wa = (data_t)0.10;
    const data_t wd = (data_t)0.025;

    int total_iterations = (NX + 1) * (VEC_COLS + 1);

    time_loop: for (int t = 0; t < TSTEPS; t++) {
        bool wr_idx = t % 2;
        bool rd_idx = !wr_idx;

        step_loop: for (int step = 0; step < total_iterations; step++) {
            #pragma HLS PIPELINE II=1
            
            int i_loop = step / (VEC_COLS + 1);
            int j_loop = step % (VEC_COLS + 1);
            
            int out_i = i_loop - 1;
            int out_j = j_loop - 1;
            int linear_idx_in = i_loop * VEC_COLS + j_loop;
            int linear_idx_out = out_i * VEC_COLS + out_j;

            // Step A: Shift Window Left by 8 elements
            for (int r = 0; r < 3; r++) {
                #pragma HLS UNROLL
                for (int c = 0; c < 16; c++) {
                    #pragma HLS UNROLL
                    window[r][c] = window[r][c + 8];
                }
            }

            // Step B: Read Data (Stream on step 0, Ping-Pong otherwise)
            uint256_dt in_val = 0;
            if (i_loop < NX && j_loop < VEC_COLS) {
                if (t == 0) {
                    in_val = in_stream.read(); 
                } else {
                    in_val = buf_work[rd_idx][linear_idx_in];
                }
            }

            uint256_dt lb1_val = 0, lb0_val = 0;
            if (j_loop < VEC_COLS) {
                lb1_val = line_buf1[j_loop];
                lb0_val = line_buf0[j_loop];
            }

            // Step C: Update Line Buffers
            if (i_loop < NX && j_loop < VEC_COLS) {
                line_buf1[j_loop] = in_val;
                line_buf0[j_loop] = lb1_val;
            }

            // Step D: Unpack Data into the right side of the window (Indices 16 to 23)
            for (int k = 0; k < TILE_FACTOR; k++) {
                #pragma HLS UNROLL
                int low = k * 32;
                int high = low + 31;
                
                ap_uint<32> val_bot = in_val.range(high, low);
                ap_uint<32> val_mid = lb1_val.range(high, low);
                ap_uint<32> val_top = lb0_val.range(high, low);

                window[2][16 + k] = *(data_t*)&val_bot;
                window[1][16 + k] = *(data_t*)&val_mid;
                window[0][16 + k] = *(data_t*)&val_top;
            }

            // Step E: Compute Output (Center vector is Indices 8 to 15)
            if (out_i >= 0 && out_i < NX && out_j >= 0 && out_j < VEC_COLS) {
                uint256_dt out_block;
                
                for (int k = 0; k < TILE_FACTOR; k++) {
                    #pragma HLS UNROLL
                    int j_real = out_j * TILE_FACTOR + k;
                    data_t result;
                    
                    if (out_i == 0 || out_i == NX - 1 || j_real == 0 || j_real == NY - 1) {
                        result = window[1][8 + k]; 
                    } else {
                        // Balanced adder trees for < 10ns timing
                        acc_t sum_axis = ((acc_t)window[0][8 + k] + (acc_t)window[2][8 + k]) +
                                         ((acc_t)window[1][8 + k - 1] + (acc_t)window[1][8 + k + 1]);
                                         
                        acc_t sum_diag = ((acc_t)window[0][8 + k - 1] + (acc_t)window[0][8 + k + 1]) +
                                         ((acc_t)window[2][8 + k - 1] + (acc_t)window[2][8 + k + 1]);
                                         
                        result = (data_t)((acc_t)wc * window[1][8 + k] + (acc_t)wa * sum_axis + (acc_t)wd * sum_diag);
                    }
                    
                    int low = k * 32;
                    int high = low + 31;
                    out_block.range(high, low) = *(ap_uint<32>*)&result;
                }

                // Step F: Write Data (Stream on final step, Ping-Pong otherwise)
                if (t == TSTEPS - 1) {
                    out_stream.write(out_block);
                } else {
                    buf_work[wr_idx][linear_idx_out] = out_block;
                }
            }
        }
    }
}

// --------------------------------------------------------
// TASK 3: Store Data with Throttled AXI Bursts
// --------------------------------------------------------
void store_data(hls::stream<uint256_dt>& in_stream, data_t A_out[NX][NY]) {
    #pragma HLS INLINE off
    uint256_dt* out_ptr = (uint256_dt*)A_out;
    
    store_loop: for (int i = 0; i < TOTAL_BLOCKS; i++) {
        #pragma HLS PIPELINE II=1
        out_ptr[i] = in_stream.read();
    }
}

// --------------------------------------------------------
// TOP KERNEL
// --------------------------------------------------------
void top_kernel(const data_t A_in[NX][NY], data_t A_out[NX][NY]) {
    // 256-bit bus width & Throttled Outstanding Requests (slashes BRAM usage)
    #pragma HLS interface m_axi port=A_in offset=slave bundle=gmem0 max_read_burst_length=256 num_read_outstanding=4 max_widen_bitwidth=256
    #pragma HLS interface m_axi port=A_out offset=slave bundle=gmem1 max_write_burst_length=256 num_write_outstanding=4 max_widen_bitwidth=256
    #pragma HLS interface s_axilite port=return

    // Depth=4 forces HLS to use tiny Shift Registers (SRLs) instead of BRAM
    hls::stream<uint256_dt> in_stream("in_stream");
    hls::stream<uint256_dt> out_stream("out_stream");
    #pragma HLS STREAM variable=in_stream depth=4
    #pragma HLS STREAM variable=out_stream depth=4

    #pragma HLS DATAFLOW

    load_data(A_in, in_stream);
    process_data(in_stream, out_stream);
    store_data(out_stream, A_out);
}
