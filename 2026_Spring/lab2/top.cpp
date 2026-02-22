#include "dcl.h"
#include <hls_stream.h>
#include <ap_int.h>

// TILE_FACTOR defines the number of pixels we want to process at once
#define TILE_FACTOR 8

// VEC_COLS is the amount of reads required per row
#define VEC_COLS (NY / TILE_FACTOR)

// TOTAL_BLOCKS is the total number of reads for the image
#define TOTAL_BLOCKS (NX * VEC_COLS)

// STAGES controls the number of pipelined compute processes occur per compute loop
#define STAGES 10

typedef ap_uint<256> uint256_dt;

void load_data(const data_t A_in[NX][NY], hls::stream<uint256_dt>& out_stream) {
    #pragma HLS INLINE off
    const uint256_dt* data_in = (const uint256_dt*)A_in;
    
    // Takes inputs to be processed
    load_loop: for (int i = 0; i < TOTAL_BLOCKS; i++) {
        #pragma HLS PIPELINE II=1

        out_stream.write(data_in[i]);
    }
}

void process_data(hls::stream<uint256_dt>& in_stream, hls::stream<uint256_dt>& out_stream) {
    #pragma HLS INLINE off
    
    // store_buf keeps intermediate results to write back into DRAM later
    uint256_dt store_buf[2][TOTAL_BLOCKS];
    #pragma HLS BIND_STORAGE variable=store_buf type=ram_2p impl=bram
    #pragma HLS ARRAY_PARTITION variable=store_buf dim=1 complete
    #pragma HLS DEPENDENCE variable=store_buf type=inter false

    // line_bufs store the last two lines/rows of pixels to compute the convolution
    uint256_dt line_buf0[STAGES][VEC_COLS];
    #pragma HLS BIND_STORAGE variable=line_buf0 type=ram_2p impl=bram
    #pragma HLS ARRAY_PARTITION variable=line_buf0 dim=1 complete

    uint256_dt line_buf1[STAGES][VEC_COLS];    
    #pragma HLS BIND_STORAGE variable=line_buf1 type=ram_2p impl=bram
    #pragma HLS ARRAY_PARTITION variable=line_buf1 dim=1 complete

    // window acts as a sliding tile that computes pixel convolution of target indices 
    data_t window[STAGES][3][24];
    #pragma HLS ARRAY_PARTITION variable=window dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=window dim=2 complete

    // Constant weights (sum to 1.0)
    const data_t wc = (data_t)0.50;
    const data_t wa = (data_t)0.10;
    const data_t wd = (data_t)0.025;

    int total_iterations = TOTAL_BLOCKS + STAGES * (VEC_COLS + 1);

    // Performs multiple passes over the image each iteration
    time_loop: for (int t = 0; t < TSTEPS; t += STAGES) {
        
        bool wr_idx = (t / STAGES) % 2;
        bool rd_idx = !wr_idx;

        step_loop: for (int step = 0; step < total_iterations; step++) {
            #pragma HLS PIPELINE II=1
            
            uint256_dt stage_link[STAGES + 1];
            #pragma HLS ARRAY_PARTITION variable=stage_link complete

            // Puts input data into stage_link for convolution calculation
            if (step < TOTAL_BLOCKS) {
                if (t == 0) // First time step, read directly from input stream
                    stage_link[0] = in_stream.read(); 
                else // For every other step, read from the store_buf that holds intermediate values
                    stage_link[0] = store_buf[rd_idx][step];
            } 
            else {
                stage_link[0] = 0; // Padding at last step
            }

            // Creates hardware copies for the N stage process
            compute_stages: for (int s = 0; s < STAGES; s++) {
                #pragma HLS UNROLL
                
                int s_step = step - s * (VEC_COLS + 1);
                int out_step = s_step - (VEC_COLS + 1);

                uint256_dt s_in_val = stage_link[s];
                uint256_dt s_out_val = 0;

                // Shifts window to the next 8 pixels
                for (int r = 0; r < 3; r++) {
                    #pragma HLS UNROLL

                    for (int c = 0; c < 16; c++) {
                        #pragma HLS UNROLL

                        window[s][r][c] = window[s][r][c + 8];
                    }
                }

                // Shifts the line buffer data to capture last 2 lines/rows of pixels
                uint256_dt lb1_val = 0, lb0_val = 0;
                if (s_step >= 0) {
                    int in_j = s_step % VEC_COLS;
                    lb1_val = line_buf1[s][in_j];
                    lb0_val = line_buf0[s][in_j];
                    
                    line_buf1[s][in_j] = s_in_val;
                    line_buf0[s][in_j] = lb1_val;
                }

                // Unpack 256-bit input from the last 2 lines/rows of pixels to store into the window
                for (int k = 0; k < TILE_FACTOR; k++) {
                    #pragma HLS UNROLL

                    int low = k * 32;
                    int high = low + 31;

                    ap_uint<32> val_bot = s_in_val.range(high, low);
                    ap_uint<32> val_mid = lb1_val.range(high, low);
                    ap_uint<32> val_top = lb0_val.range(high, low);
                    
                    window[s][2][16 + k] = *(data_t*)&val_bot;
                    window[s][1][16 + k] = *(data_t*)&val_mid;
                    window[s][0][16 + k] = *(data_t*)&val_top;
                }

                // Computes convolution parallely using pixels in the window 
                if (out_step >= 0 && out_step < TOTAL_BLOCKS) {

                    // Calculates the row and column indices from the current step value
                    int out_i = out_step / VEC_COLS;
                    int out_j = out_step % VEC_COLS;

                    for (int k = 0; k < TILE_FACTOR; k++) {
                        #pragma HLS UNROLL

                        int j_real = out_j * TILE_FACTOR + k;
                        data_t result;
                        
                        if (out_i == 0 || out_i == NX - 1 || j_real == 0 || j_real == NY - 1) {
                            result = window[s][1][8 + k]; 
                        } else {
                            acc_t sum_ax = (acc_t)window[s][0][8 + k] + (acc_t)window[s][2][8 + k] + 
                                           (acc_t)window[s][1][8 + k - 1] + (acc_t)window[s][1][8 + k + 1];
                                           
                            acc_t sum_dg = (acc_t)window[s][0][8 + k - 1] + (acc_t)window[s][0][8 + k + 1] + 
                                           (acc_t)window[s][2][8 + k - 1] + (acc_t)window[s][2][8 + k + 1];
                                           
                            result = (data_t)((acc_t)wc * window[s][1][8 + k] + (acc_t)wa * sum_ax + (acc_t)wd * sum_dg);
                        }
                        
                        int low = k * 32, high = low + 31;
                        s_out_val.range(high, low) = *(ap_uint<32>*)&result;
                    }
                }
                
                // Stores calculated to eb written to output
                stage_link[s + 1] = s_out_val;
            }

            int final_out_step = step - STAGES * (VEC_COLS + 1);

            // Writes result into output
            if (final_out_step >= 0 && final_out_step < TOTAL_BLOCKS) {
                if (t == TSTEPS - STAGES) {
                    out_stream.write(stage_link[STAGES]);
                } 
                else {
                    store_buf[wr_idx][final_out_step] = stage_link[STAGES];
                }
            }
        }
    }
}

void store_data(hls::stream<uint256_dt>& in_stream, data_t A_out[NX][NY]) {
    #pragma HLS INLINE off
    uint256_dt* data_out = (uint256_dt*)A_out;
    
    // Stores results into output
    store_loop: for (int i = 0; i < TOTAL_BLOCKS; i++) {
        #pragma HLS PIPELINE II=1
        data_out[i] = in_stream.read();
    }
}

void top_kernel(const data_t A_in[NX][NY], data_t A_out[NX][NY]) {
    #pragma HLS interface m_axi port=A_in offset=slave bundle=gmem0 max_read_burst_length=256 num_read_outstanding=4 max_widen_bitwidth=256
    #pragma HLS interface m_axi port=A_out offset=slave bundle=gmem1 max_write_burst_length=256 num_write_outstanding=4 max_widen_bitwidth=256
    #pragma HLS interface s_axilite port=return

    hls::stream<uint256_dt> in_stream("in_stream");
    #pragma HLS STREAM variable=in_stream depth=4

    hls::stream<uint256_dt> out_stream("out_stream");
    #pragma HLS STREAM variable=out_stream depth=4

    #pragma HLS DATAFLOW

    load_data(A_in, in_stream);
    process_data(in_stream, out_stream);
    store_data(out_stream, A_out);
}
