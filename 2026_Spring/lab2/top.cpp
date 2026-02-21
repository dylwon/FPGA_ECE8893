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
    
    uint256_dt buf_work[2][TOTAL_BLOCKS];
    #pragma HLS BIND_STORAGE variable=buf_work type=ram_2p impl=bram
    #pragma HLS ARRAY_PARTITION variable=buf_work dim=1 complete
    #pragma HLS DEPENDENCE variable=buf_work type=inter false

    // Line Buffers and Window for STAGE A (Time Step t)
    uint256_dt line_buf0_A[VEC_COLS], line_buf1_A[VEC_COLS];
    #pragma HLS BIND_STORAGE variable=line_buf0_A type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=line_buf1_A type=ram_2p impl=bram
    data_t winA[3][24];
    #pragma HLS ARRAY_PARTITION variable=winA dim=0 complete

    // Line Buffers and Window for STAGE B (Time Step t+1)
    uint256_dt line_buf0_B[VEC_COLS], line_buf1_B[VEC_COLS];
    #pragma HLS BIND_STORAGE variable=line_buf0_B type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=line_buf1_B type=ram_2p impl=bram
    data_t winB[3][24];
    #pragma HLS ARRAY_PARTITION variable=winB dim=0 complete

    const data_t wc = (data_t)0.50, wa = (data_t)0.10, wd = (data_t)0.025;

    // +2 iterations required to flush the pipeline for both stages
    int total_iterations = (NX + 2) * (VEC_COLS + 2);

    // Notice we skip by 2! We process two time steps per pass.
    time_loop: for (int t = 0; t < TSTEPS; t += 2) {
        bool wr_idx = (t / 2) % 2;
        bool rd_idx = !wr_idx;

        step_loop: for (int step = 0; step < total_iterations; step++) {
            #pragma HLS PIPELINE II=1
            
            int i_loop = step / (VEC_COLS + 2);
            int j_loop = step % (VEC_COLS + 2);
            
            int s1_i = i_loop - 1, s1_j = j_loop - 1; // Stage A Output Coordinates
            int s2_i = i_loop - 2, s2_j = j_loop - 2; // Stage B Output Coordinates

            // ==========================================
            // STAGE A: Computes Time Step t
            // ==========================================
            for (int r = 0; r < 3; r++) {
                #pragma HLS UNROLL
                for (int c = 0; c < 16; c++) {
                    #pragma HLS UNROLL
                    winA[r][c] = winA[r][c + 8];
                }
            }

            uint256_dt in_val = 0;
            if (i_loop < NX && j_loop < VEC_COLS) {
                if (t == 0) in_val = in_stream.read(); 
                else        in_val = buf_work[rd_idx][i_loop * VEC_COLS + j_loop];
            }

            uint256_dt lb1A_val = 0, lb0A_val = 0;
            if (j_loop < VEC_COLS) {
                lb1A_val = line_buf1_A[j_loop];
                lb0A_val = line_buf0_A[j_loop];
            }
            if (i_loop < NX && j_loop < VEC_COLS) {
                line_buf1_A[j_loop] = in_val;
                line_buf0_A[j_loop] = lb1A_val;
            }

            for (int k = 0; k < TILE_FACTOR; k++) {
                #pragma HLS UNROLL
                int low = k * 32, high = low + 31;
                ap_uint<32> val_bot = in_val.range(high, low);
                ap_uint<32> val_mid = lb1A_val.range(high, low);
                ap_uint<32> val_top = lb0A_val.range(high, low);
                winA[2][16 + k] = *(data_t*)&val_bot;
                winA[1][16 + k] = *(data_t*)&val_mid;
                winA[0][16 + k] = *(data_t*)&val_top;
            }

            uint256_dt s1_out = 0;
            if (s1_i >= 0 && s1_i < NX && s1_j >= 0 && s1_j < VEC_COLS) {
                for (int k = 0; k < TILE_FACTOR; k++) {
                    #pragma HLS UNROLL
                    int j_real = s1_j * TILE_FACTOR + k;
                    data_t result;
                    if (s1_i == 0 || s1_i == NX - 1 || j_real == 0 || j_real == NY - 1) {
                        result = winA[1][8 + k]; 
                    } else {
                        acc_t sum_ax = (acc_t)winA[0][8+k] + (acc_t)winA[2][8+k] + (acc_t)winA[1][8+k-1] + (acc_t)winA[1][8+k+1];
                        acc_t sum_dg = (acc_t)winA[0][8+k-1] + (acc_t)winA[0][8+k+1] + (acc_t)winA[2][8+k-1] + (acc_t)winA[2][8+k+1];
                        result = (data_t)((acc_t)wc * winA[1][8+k] + (acc_t)wa * sum_ax + (acc_t)wd * sum_dg);
                    }
                    int low = k * 32, high = low + 31;
                    s1_out.range(high, low) = *(ap_uint<32>*)&result;
                }
            }

            // ==========================================
            // STAGE B: Consumes Stage A, Computes t+1
            // ==========================================
            for (int r = 0; r < 3; r++) {
                #pragma HLS UNROLL
                for (int c = 0; c < 16; c++) {
                    #pragma HLS UNROLL
                    winB[r][c] = winB[r][c + 8];
                }
            }

            uint256_dt s2_in_val = s1_out; 
            uint256_dt lb1B_val = 0, lb0B_val = 0;
            
            if (s1_j >= 0 && s1_j < VEC_COLS) {
                lb1B_val = line_buf1_B[s1_j];
                lb0B_val = line_buf0_B[s1_j];
            }
            if (s1_i >= 0 && s1_i < NX && s1_j >= 0 && s1_j < VEC_COLS) {
                line_buf1_B[s1_j] = s2_in_val;
                line_buf0_B[s1_j] = lb1B_val;
            }

            for (int k = 0; k < TILE_FACTOR; k++) {
                #pragma HLS UNROLL
                int low = k * 32, high = low + 31;
                ap_uint<32> val_bot = s2_in_val.range(high, low);
                ap_uint<32> val_mid = lb1B_val.range(high, low);
                ap_uint<32> val_top = lb0B_val.range(high, low);
                winB[2][16 + k] = *(data_t*)&val_bot;
                winB[1][16 + k] = *(data_t*)&val_mid;
                winB[0][16 + k] = *(data_t*)&val_top;
            }

            if (s2_i >= 0 && s2_i < NX && s2_j >= 0 && s2_j < VEC_COLS) {
                uint256_dt out_block;
                for (int k = 0; k < TILE_FACTOR; k++) {
                    #pragma HLS UNROLL
                    int j_real = s2_j * TILE_FACTOR + k;
                    data_t result;
                    if (s2_i == 0 || s2_i == NX - 1 || j_real == 0 || j_real == NY - 1) {
                        result = winB[1][8 + k]; 
                    } else {
                        acc_t sum_ax = (acc_t)winB[0][8+k] + (acc_t)winB[2][8+k] + (acc_t)winB[1][8+k-1] + (acc_t)winB[1][8+k+1];
                        acc_t sum_dg = (acc_t)winB[0][8+k-1] + (acc_t)winB[0][8+k+1] + (acc_t)winB[2][8+k-1] + (acc_t)winB[2][8+k+1];
                        result = (data_t)((acc_t)wc * winB[1][8+k] + (acc_t)wa * sum_ax + (acc_t)wd * sum_dg);
                    }
                    int low = k * 32, high = low + 31;
                    out_block.range(high, low) = *(ap_uint<32>*)&result;
                }

                if (t == TSTEPS - 2) {
                    out_stream.write(out_block);
                } else {
                    buf_work[wr_idx][s2_i * VEC_COLS + s2_j] = out_block;
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
