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
// TASK 2: Parameterized N-Stage Systolic Compute Engine
// --------------------------------------------------------
#define STAGES 10

void process_data(hls::stream<uint256_dt>& in_stream, hls::stream<uint256_dt>& out_stream) {
    #pragma HLS INLINE off
    
    uint256_dt buf_work[2][TOTAL_BLOCKS];
    #pragma HLS BIND_STORAGE variable=buf_work type=ram_2p impl=bram
    #pragma HLS ARRAY_PARTITION variable=buf_work dim=1 complete
    #pragma HLS DEPENDENCE variable=buf_work type=inter false

    uint256_dt line_buf0[STAGES][VEC_COLS];
    uint256_dt line_buf1[STAGES][VEC_COLS];
    #pragma HLS BIND_STORAGE variable=line_buf0 type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=line_buf1 type=ram_2p impl=bram
    #pragma HLS ARRAY_PARTITION variable=line_buf0 dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=line_buf1 dim=1 complete

    data_t window[STAGES][3][24];
    #pragma HLS ARRAY_PARTITION variable=window dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=window dim=2 complete

    const data_t wc = (data_t)0.50, wa = (data_t)0.10, wd = (data_t)0.025;
    int total_iterations = TOTAL_BLOCKS + STAGES * (VEC_COLS + 1);

    time_loop: for (int t = 0; t < TSTEPS; t += STAGES) {
        
        bool wr_idx = (t / STAGES) % 2;
        bool rd_idx = !wr_idx;

        step_loop: for (int step = 0; step < total_iterations; step++) {
            #pragma HLS PIPELINE II=1
            
            uint256_dt stage_link[STAGES + 1];
            #pragma HLS ARRAY_PARTITION variable=stage_link complete

            // 1. Read Input Data
            if (step < TOTAL_BLOCKS) {
                if (t == 0) stage_link[0] = in_stream.read(); 
                else        stage_link[0] = buf_work[rd_idx][step];
            } else {
                stage_link[0] = 0; // Padding during flush
            }

            // 2. Unroll Compute Logic
            compute_stages: for (int s = 0; s < STAGES; s++) {
                #pragma HLS UNROLL
                
                int s_step = step - s * (VEC_COLS + 1);
                int out_step = s_step - (VEC_COLS + 1);

                uint256_dt s_in_val = stage_link[s];
                uint256_dt s_out_val = 0;

                // Shift Window
                for (int r = 0; r < 3; r++) {
                    #pragma HLS UNROLL
                    for (int c = 0; c < 16; c++) {
                        #pragma HLS UNROLL
                        window[s][r][c] = window[s][r][c + 8];
                    }
                }

                // Update Line Buffers
                uint256_dt lb1_val = 0, lb0_val = 0;
                // THE FIX: Allow line buffers to shift during the flush phase!
                if (s_step >= 0) {
                    int in_j = s_step % VEC_COLS;
                    lb1_val = line_buf1[s][in_j];
                    lb0_val = line_buf0[s][in_j];
                    
                    line_buf1[s][in_j] = s_in_val; // Pushes 0s automatically during flush
                    line_buf0[s][in_j] = lb1_val;
                }

                // Unpack Data into Window
                for (int k = 0; k < TILE_FACTOR; k++) {
                    #pragma HLS UNROLL
                    int low = k * 32, high = low + 31;
                    ap_uint<32> val_bot = s_in_val.range(high, low);
                    ap_uint<32> val_mid = lb1_val.range(high, low);
                    ap_uint<32> val_top = lb0_val.range(high, low);
                    
                    window[s][2][16 + k] = *(data_t*)&val_bot;
                    window[s][1][16 + k] = *(data_t*)&val_mid;
                    window[s][0][16 + k] = *(data_t*)&val_top;
                }

                // Compute Output
                if (out_step >= 0 && out_step < TOTAL_BLOCKS) {
                    int out_i = out_step / VEC_COLS;
                    int out_j = out_step % VEC_COLS;

                    for (int k = 0; k < TILE_FACTOR; k++) {
                        #pragma HLS UNROLL
                        int j_real = out_j * TILE_FACTOR + k;
                        data_t result;
                        
                        if (out_i == 0 || out_i == NX - 1 || j_real == 0 || j_real == NY - 1) {
                            result = window[s][1][8 + k]; 
                        } else {
                            acc_t sum_ax = (acc_t)window[s][0][8+k] + (acc_t)window[s][2][8+k] + 
                                           (acc_t)window[s][1][8+k-1] + (acc_t)window[s][1][8+k+1];
                                           
                            acc_t sum_dg = (acc_t)window[s][0][8+k-1] + (acc_t)window[s][0][8+k+1] + 
                                           (acc_t)window[s][2][8+k-1] + (acc_t)window[s][2][8+k+1];
                                           
                            result = (data_t)((acc_t)wc * window[s][1][8+k] + (acc_t)wa * sum_ax + (acc_t)wd * sum_dg);
                        }
                        
                        int low = k * 32, high = low + 31;
                        s_out_val.range(high, low) = *(ap_uint<32>*)&result;
                    }
                }
                
                stage_link[s + 1] = s_out_val;
            }

            // 3. Write Output Data
            int final_out_step = step - STAGES * (VEC_COLS + 1);
            if (final_out_step >= 0 && final_out_step < TOTAL_BLOCKS) {
                if (t == TSTEPS - STAGES) {
                    out_stream.write(stage_link[STAGES]);
                } else {
                    buf_work[wr_idx][final_out_step] = stage_link[STAGES];
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
