#include "dcl.h"
#include <hls_stream.h>
#include <ap_int.h>

// Maximize AXI4 Bus Width: 512 bits = 16 * 32-bit elements
#define TILE_FACTOR 16
#define VEC_COLS (NY / TILE_FACTOR)
#define TOTAL_BLOCKS (NX * VEC_COLS)

typedef ap_uint<512> uint512_dt;

// --------------------------------------------------------
// TASK 1: Burst Read 512-bit Vectors from DDR
// --------------------------------------------------------
void load_data(const data_t A_in[NX][NY], hls::stream<uint512_dt>& out_stream) {
    #pragma HLS INLINE off
    const uint512_dt* in_ptr = (const uint512_dt*)A_in;
    
    load_loop: for (int i = 0; i < TOTAL_BLOCKS; i++) {
        #pragma HLS PIPELINE II=1
        out_stream.write(in_ptr[i]);
    }
}

// --------------------------------------------------------
// TASK 2: 30-Step Compute Engine (16-wide Vectorization)
// --------------------------------------------------------
void process_data(hls::stream<uint512_dt>& in_stream, hls::stream<uint512_dt>& out_stream) {
    #pragma HLS INLINE off
    
    uint512_dt buf_work[2][TOTAL_BLOCKS];
    #pragma HLS BIND_STORAGE variable=buf_work type=ram_2p impl=bram

    uint512_dt line_buf0[VEC_COLS];
    uint512_dt line_buf1[VEC_COLS];
    #pragma HLS BIND_STORAGE variable=line_buf0 type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=line_buf1 type=ram_2p impl=bram

    // Widened 3x48 Window (Prev 16, Curr 16, Next 16)
    data_t window[3][48];
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

            // Step A: Shift Window Left by 16 elements
            for (int r = 0; r < 3; r++) {
                #pragma HLS UNROLL
                for (int c = 0; c < 32; c++) {
                    #pragma HLS UNROLL
                    window[r][c] = window[r][c + 16];
                }
            }

            // Step B: Read Data (Stream on first step, internal Ping-Pong otherwise)
            uint512_dt in_val = 0;
            if (i_loop < NX && j_loop < VEC_COLS) {
                if (t == 0) {
                    in_val = in_stream.read(); 
                } else {
                    in_val = buf_work[rd_idx][linear_idx_in];
                }
            }

            uint512_dt lb1_val = 0, lb0_val = 0;
            if (j_loop < VEC_COLS) {
                lb1_val = line_buf1[j_loop];
                lb0_val = line_buf0[j_loop];
            }

            // Step C: Update Line Buffers
            if (i_loop < NX && j_loop < VEC_COLS) {
                line_buf1[j_loop] = in_val;
                line_buf0[j_loop] = lb1_val;
            }

            // Step D: Unpack into the right side of the window (Indices 32 to 47)
            for (int k = 0; k < TILE_FACTOR; k++) {
                #pragma HLS UNROLL
                int low = k * 32;
                int high = low + 31;
                
                ap_uint<32> val_bot = in_val.range(high, low);
                ap_uint<32> val_mid = lb1_val.range(high, low);
                ap_uint<32> val_top = lb0_val.range(high, low);

                window[2][32 + k] = *(data_t*)&val_bot;
                window[1][32 + k] = *(data_t*)&val_mid;
                window[0][32 + k] = *(data_t*)&val_top;
            }

            // Step E: Compute Output (Center vector is Indices 16 to 31)
            if (out_i >= 0 && out_i < NX && out_j >= 0 && out_j < VEC_COLS) {
                uint512_dt out_block;
                
                for (int k = 0; k < TILE_FACTOR; k++) {
                    #pragma HLS UNROLL
                    int j_real = out_j * TILE_FACTOR + k;
                    data_t result;
                    
                    if (out_i == 0 || out_i == NX - 1 || j_real == 0 || j_real == NY - 1) {
                        result = window[1][16 + k]; 
                    } else {
                        acc_t sum_axis = ((acc_t)window[0][16 + k] + (acc_t)window[2][16 + k]) +
                                         ((acc_t)window[1][16 + k - 1] + (acc_t)window[1][16 + k + 1]);
                                         
                        acc_t sum_diag = ((acc_t)window[0][16 + k - 1] + (acc_t)window[0][16 + k + 1]) +
                                         ((acc_t)window[2][16 + k - 1] + (acc_t)window[2][16 + k + 1]);
                                         
                        result = (data_t)((acc_t)wc * window[1][16 + k] + (acc_t)wa * sum_axis + (acc_t)wd * sum_diag);
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
// TASK 3: Burst Write 512-bit Vectors to DDR
// --------------------------------------------------------
void store_data(hls::stream<uint512_dt>& in_stream, data_t A_out[NX][NY]) {
    #pragma HLS INLINE off
    uint512_dt* out_ptr = (uint512_dt*)A_out;
    
    store_loop: for (int i = 0; i < TOTAL_BLOCKS; i++) {
        #pragma HLS PIPELINE II=1
        out_ptr[i] = in_stream.read();
    }
}

// --------------------------------------------------------
// TOP KERNEL
// --------------------------------------------------------
void top_kernel(const data_t A_in[NX][NY], data_t A_out[NX][NY]) {
    // AXI Memory Mapped Interfaces for DDR
    #pragma HLS interface m_axi port=A_in offset=slave bundle=gmem0 max_read_burst_length=256 max_widen_bitwidth=512
    #pragma HLS interface m_axi port=A_out offset=slave bundle=gmem1 max_write_burst_length=256 max_widen_bitwidth=512
    #pragma HLS interface s_axilite port=return

    // Internal FIFOs for Task-Level Pipelining
    hls::stream<uint512_dt> in_stream("in_stream");
    hls::stream<uint512_dt> out_stream("out_stream");
    #pragma HLS STREAM variable=in_stream depth=16
    #pragma HLS STREAM variable=out_stream depth=16

    #pragma HLS DATAFLOW

    load_data(A_in, in_stream);
    process_data(in_stream, out_stream);
    store_data(out_stream, A_out);
}
