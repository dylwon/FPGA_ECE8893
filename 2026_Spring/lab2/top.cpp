#include "dcl.h"
#include <hls_stream.h>
#include <ap_int.h>

#define TILE_FACTOR 8

typedef ap_uint<256> uint256_dt;

void read_input(const data_t A_in[NX][NY], hls::stream<uint256_dt>& in_stream) {
    #pragma HLS INLINE off

    const uint256_dt* in_ptr = (const uint256_dt*)A_in;
    int total_blocks = (NX * NY) / TILE_FACTOR;

    read_loop: for (int i = 0; i < total_blocks; i++) {
        #pragma HLS PIPELINE II=1
        in_stream.write(in_ptr[i]);
    }
}

void compute_stencil(hls::stream<uint256_dt>& in_stream, hls::stream<uint256_dt>& out_stream) {
    #pragma HLS INLINE off

    static data_t buf0[NX][NY];
    static data_t buf1[NX][NY];

    #pragma HLS ARRAY_PARTITION variable=buf0 dim=1 cyclic factor=4
    #pragma HLS ARRAY_PARTITION variable=buf0 dim=2 cyclic factor=8

    #pragma HLS ARRAY_PARTITION variable=buf1 dim=1 cyclic factor=4
    #pragma HLS ARRAY_PARTITION variable=buf1 dim=2 cyclic factor=8

    const data_t wc = (data_t)0.50;
    const data_t wa = (data_t)0.10;
    const data_t wd = (data_t)0.025;

    load_row: for (int i = 0; i < NX; i++) {
        load_col: for (int j = 0; j < NY; j += TILE_FACTOR) {
            #pragma HLS PIPELINE II=1
            
            uint256_dt block_val = in_stream.read();
            
            load_inner: for (int jj = 0; jj < TILE_FACTOR; jj++) {
                #pragma HLS UNROLL
                int low = jj * 32;
                int high = low + 31;
                ap_uint<32> temp_int = block_val.range(high, low);
                buf0[i][j + jj] = *(data_t*)&temp_int;
            }
        }
    }

    time_loop: for (int t = 0; t < TSTEPS; t++) {
        
        if (t % 2 == 0) {
            boundary_row_even: for (int j = 0; j < NY; j += TILE_FACTOR) {
                #pragma HLS PIPELINE II=1
                for (int jj = 0; jj < TILE_FACTOR; jj++) {
                    #pragma HLS UNROLL
                    buf1[0][j + jj] = buf0[0][j + jj];
                    buf1[NX - 1][j + jj] = buf0[NX - 1][j + jj];
                }   
            }

            boundary_col_even: for (int i = 0; i < NX; i += TILE_FACTOR) {
                #pragma HLS PIPELINE II=1
                for (int ii = 0; ii < TILE_FACTOR; ii++) {
                    #pragma HLS UNROLL
                    buf1[i + ii][0] = buf0[i + ii][0];
                    buf1[i + ii][NY - 1] = buf0[i + ii][NY - 1];
                }
            }

            compute_i_even: for (int i = 1; i < NX - 1; i++) {
                compute_j_even: for (int j = 0; j < NY; j += TILE_FACTOR) {
                    #pragma HLS PIPELINE II=1

                    for (int jj = 0; jj < TILE_FACTOR; jj++) {
                        #pragma HLS UNROLL

                        if (j + jj > 0 && j + jj < NY - 1) {
                            int j_idx = j + jj;
                            acc_t sum_axis = (acc_t)buf0[i - 1][j_idx] + (acc_t)buf0[i + 1][j_idx] +
                                             (acc_t)buf0[i][j_idx - 1] + (acc_t)buf0[i][j_idx + 1];
                            acc_t sum_diag = (acc_t)buf0[i - 1][j_idx - 1] + (acc_t)buf0[i - 1][j_idx + 1] +
                                             (acc_t)buf0[i + 1][j_idx - 1] + (acc_t)buf0[i + 1][j_idx + 1];
                            acc_t out = (acc_t)wc * buf0[i][j_idx] + (acc_t)wa * sum_axis + (acc_t)wd * sum_diag;
                            buf1[i][j_idx] = (data_t)out;
                        }
                    } 
                }
            }
        }
        else {
            boundary_row_odd: for (int j = 0; j < NY; j += TILE_FACTOR) {
                #pragma HLS PIPELINE II=1
                for (int jj = 0; jj < TILE_FACTOR; jj++) {
                    #pragma HLS UNROLL
                    buf0[0][j + jj] = buf1[0][j + jj];
                    buf0[NX - 1][j + jj] = buf1[NX - 1][j + jj];
                }   
            }

            boundary_col_odd: for (int i = 0; i < NX; i += TILE_FACTOR) {
                #pragma HLS PIPELINE II=1
                for (int ii = 0; ii < TILE_FACTOR; ii++) {
                    #pragma HLS UNROLL
                    buf0[i + ii][0] = buf1[i + ii][0];
                    buf0[i + ii][NY - 1] = buf1[i + ii][NY - 1];
                }
            }

            compute_i_odd: for (int i = 1; i < NX - 1; i++) {
                compute_j_odd: for (int j = 0; j < NY; j += TILE_FACTOR) {
                    #pragma HLS PIPELINE II=1

                    for (int jj = 0; jj < TILE_FACTOR; jj++) {
                        #pragma HLS UNROLL

                        if (j + jj > 0 && j + jj < NY - 1) {
                            int j_idx = j + jj;
                            acc_t sum_axis = (acc_t)buf1[i - 1][j_idx] + (acc_t)buf1[i + 1][j_idx] +
                                             (acc_t)buf1[i][j_idx - 1] + (acc_t)buf1[i][j_idx + 1];
                            acc_t sum_diag = (acc_t)buf1[i - 1][j_idx - 1] + (acc_t)buf1[i - 1][j_idx + 1] +
                                             (acc_t)buf1[i + 1][j_idx - 1] + (acc_t)buf1[i + 1][j_idx + 1];
                            acc_t out = (acc_t)wc * buf1[i][j_idx] + (acc_t)wa * sum_axis + (acc_t)wd * sum_diag;
                            buf0[i][j_idx] = (data_t)out;
                        }
                    } 
                }
            }
        }
    }

    bool buf1_select = (TSTEPS % 2 != 0);
    
    store_row: for (int i = 0; i < NX; i++) {
        store_col: for (int j = 0; j < NY; j += TILE_FACTOR) {
            #pragma HLS PIPELINE II=1
            
            uint256_dt block_val;
            
            store_inner: for (int jj = 0; jj < TILE_FACTOR; jj++) {
                #pragma HLS UNROLL
                data_t temp = buf1_select ? buf1[i][j + jj] : buf0[i][j + jj];
                int low = jj * 32;
                int high = low + 31;
                block_val.range(high, low) = *(ap_uint<32>*)&temp;
            }
            out_stream.write(block_val);
        }
    }
}

void write_output(hls::stream<uint256_dt>& out_stream, data_t A_out[NX][NY]) {
    #pragma HLS INLINE off

    uint256_dt* out_ptr = (uint256_dt*)A_out;
    int total_blocks = (NX * NY) / TILE_FACTOR;

    write_loop: for (int i = 0; i < total_blocks; i++) {
        #pragma HLS PIPELINE II=1
        out_ptr[i] = out_stream.read();
    }
}

void top_kernel(const data_t A_in[NX][NY], data_t A_out[NX][NY]) {
    #pragma HLS interface m_axi port=A_in offset=slave bundle=A_in max_read_burst_length=256 num_read_outstanding=32
    #pragma HLS interface m_axi port=A_out offset=slave bundle=A_out max_write_burst_length=256 num_write_outstanding=32
    #pragma HLS interface s_axilite port=return

    #pragma HLS DATAFLOW
    
    hls::stream<uint256_dt> in_stream("input_stream");
    hls::stream<uint256_dt> out_stream("output_stream");
    
    #pragma HLS STREAM variable=in_stream depth=64
    #pragma HLS STREAM variable=out_stream depth=64

    read_input(A_in, in_stream);
    compute_stencil(in_stream, out_stream);
    write_output(out_stream, A_out);
}
