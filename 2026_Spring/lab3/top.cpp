#include "dcl.h"
#include <hls_stream.h>
#include <ap_int.h>

#define TILE_FACTOR 8
#define VEC_N (N / TILE_FACTOR)
#define VEC_BLOCK (BLOCK / TILE_FACTOR)

typedef ap_uint<256> uint256_dt;

static inline data_t abs_fp(data_t x) {
    #pragma HLS INLINE
    return (x < (data_t)0) ? (data_t)(-x) : x;
}

static inline data_t clamp_fp(data_t x, data_t lo, data_t hi) {
    #pragma HLS INLINE
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

// --------------------------------------------------------
// K0: Load, Preprocess, and Split Stream
// --------------------------------------------------------
void load_and_K0(const data_t in[N], hls::stream<uint256_dt>& out_k1, hls::stream<uint256_dt>& out_k2) {
    #pragma HLS INLINE off
    const coef_t alpha = (coef_t)0.875;
    const coef_t beta  = (coef_t)0.125;

    load_loop: for (int i = 0; i < VEC_N; i++) {
        #pragma HLS PIPELINE II=1
        uint256_dt out_block = 0;

        for (int k = 0; k < TILE_FACTOR; k++) {
            #pragma HLS UNROLL
            data_t val = in[i * TILE_FACTOR + k];
            data_t s0_val = (data_t)((acc_t)alpha * (acc_t)val + (acc_t)beta);

            ap_uint<32> tmp_out = s0_val.range(); 
            out_block(k * 32 + 31, k * 32) = tmp_out;
        }
        out_k1.write(out_block);
        out_k2.write(out_block);
    }
}

// --------------------------------------------------------
// K1: Transform (Branchless 1D Sliding Window)
// --------------------------------------------------------
void K1_transform(hls::stream<uint256_dt>& in_stream, hls::stream<uint256_dt>& out_stream) {
    #pragma HLS INLINE off
    const coef_t w0 = (coef_t)0.50;
    const coef_t w1 = (coef_t)(-0.25);
    const coef_t w2 = (coef_t)0.125;

    data_t delay_reg[2] = {0, 0};
    #pragma HLS ARRAY_PARTITION variable=delay_reg complete

    k1_loop: for (int i = 0; i < VEC_N; i++) {
        #pragma HLS PIPELINE II=1
        uint256_dt block = in_stream.read();
        uint256_dt out_block = 0;

        data_t window[TILE_FACTOR + 2];
        #pragma HLS ARRAY_PARTITION variable=window complete

        window[0] = delay_reg[0];
        window[1] = delay_reg[1];

        for (int k = 0; k < TILE_FACTOR; k++) {
            #pragma HLS UNROLL
            ap_uint<32> tmp = block(k * 32 + 31, k * 32);
            data_t val;
            val.range() = tmp;
            window[k + 2] = val;
        }

        delay_reg[0] = window[TILE_FACTOR];
        delay_reg[1] = window[TILE_FACTOR + 1];

        for (int k = 0; k < TILE_FACTOR; k++) {
            #pragma HLS UNROLL
            data_t x0 = window[k + 2];
            data_t x1 = window[k + 1];
            data_t x2 = window[k];

            acc_t acc = (acc_t)w0 * (acc_t)x0 + (acc_t)w1 * (acc_t)x1 + (acc_t)w2 * (acc_t)x2;
            data_t y = clamp_fp(abs_fp((data_t)acc), (data_t)0, (data_t)7.5);

            ap_uint<32> tmp_out = y.range();
            out_block(k * 32 + 31, k * 32) = tmp_out;
        }
        out_stream.write(out_block);
    }
}

// --------------------------------------------------------
// K2: Per-Block Statistic Accumulator (FLATTENED)
// --------------------------------------------------------
void K2_statistics(hls::stream<uint256_dt>& in_stream, hls::stream<stat_t>& stat_stream) {
    #pragma HLS INLINE off
    const stat_t eps = (stat_t)0.5;
    acc_t sum_abs = 0;

    // Single continuous loop prevents pipeline drains!
    k2_loop: for (int i = 0; i < VEC_N; i++) {
        #pragma HLS PIPELINE II=1
        uint256_dt block = in_stream.read();
        acc_t local_sum = 0;

        for (int k = 0; k < TILE_FACTOR; k++) {
            #pragma HLS UNROLL
            ap_uint<32> tmp = block(k * 32 + 31, k * 32);
            data_t val;
            val.range() = tmp;
            local_sum += (acc_t)abs_fp(val);
        }
        
        sum_abs += local_sum;

        // At the exact end of a block, compute stat and reset
        if ((i + 1) % VEC_BLOCK == 0) {
            stat_t avg_abs = (stat_t)(sum_abs / (acc_t)BLOCK);
            stat_stream.write(avg_abs + eps);
            sum_abs = 0; 
        }
    }
}

// --------------------------------------------------------
// K3: Join and Normalize (FLATTENED)
// --------------------------------------------------------
void K3_normalize(hls::stream<uint256_dt>& in_stream, hls::stream<stat_t>& stat_stream, hls::stream<uint256_dt>& out_stream) {
    #pragma HLS INLINE off
    stat_t inv_st = 0;

    // Single continuous loop
    k3_loop: for (int i = 0; i < VEC_N; i++) {
        #pragma HLS PIPELINE II=1
        
        // Fetch new statistic and divide ONLY at the start of a new block
        if (i % VEC_BLOCK == 0) {
            stat_t st = stat_stream.read();
            inv_st = (stat_t)((acc_t)1 / (acc_t)st);
        }

        uint256_dt block = in_stream.read();
        uint256_dt out_block = 0;

        for (int k = 0; k < TILE_FACTOR; k++) {
            #pragma HLS UNROLL
            ap_uint<32> tmp = block(k * 32 + 31, k * 32);
            data_t val;
            val.range() = tmp;

            data_t norm_val = (data_t)((acc_t)val * (acc_t)inv_st);

            ap_uint<32> tmp_out = norm_val.range();
            out_block(k * 32 + 31, k * 32) = tmp_out;
        }
        out_stream.write(out_block);
    }
}

// --------------------------------------------------------
// K4: Postprocess and Store
// --------------------------------------------------------
void K4_and_store(hls::stream<uint256_dt>& in_stream, data_t out[N]) {
    #pragma HLS INLINE off
    const coef_t gamma = (coef_t)1.25;
    const coef_t delta = (coef_t)0.05;

    k4_loop: for (int i = 0; i < VEC_N; i++) {
        #pragma HLS PIPELINE II=1
        uint256_dt block = in_stream.read();

        for (int k = 0; k < TILE_FACTOR; k++) {
            #pragma HLS UNROLL
            ap_uint<32> tmp = block(k * 32 + 31, k * 32);
            data_t val;
            val.range() = tmp;

            data_t z = (data_t)((acc_t)gamma * (acc_t)val + (acc_t)delta);
            out[i * TILE_FACTOR + k] = clamp_fp(z, (data_t)0, (data_t)7.9);
        }
    }
}

// --------------------------------------------------------
// TOP KERNEL
// --------------------------------------------------------
void top_kernel(const data_t in[N], data_t out[N]) {
    #pragma HLS interface m_axi port=in offset=slave bundle=gmem0 max_read_burst_length=256 num_read_outstanding=4 max_widen_bitwidth=256
    #pragma HLS interface m_axi port=out offset=slave bundle=gmem1 max_write_burst_length=256 num_write_outstanding=4 max_widen_bitwidth=256
    #pragma HLS interface s_axilite port=return

    hls::stream<uint256_dt> stream_s0_to_k1("stream_s0_to_k1");
    #pragma HLS STREAM variable=stream_s0_to_k1 depth=16

    hls::stream<uint256_dt> stream_s0_to_k2("stream_s0_to_k2");
    #pragma HLS STREAM variable=stream_s0_to_k2 depth=16

    // THE SHOCK ABSORBER: Depth of 256 easily absorbs the 30-cycle division latency plus multiple blocks.
    hls::stream<uint256_dt> stream_s1("stream_s1");
    #pragma HLS STREAM variable=stream_s1 depth=256

    hls::stream<stat_t> stream_stat("stream_stat");
    #pragma HLS STREAM variable=stream_stat depth=16

    hls::stream<uint256_dt> stream_s3("stream_s3");
    #pragma HLS STREAM variable=stream_s3 depth=16

    #pragma HLS DATAFLOW

    load_and_K0(in, stream_s0_to_k1, stream_s0_to_k2);
    K1_transform(stream_s0_to_k1, stream_s1);
    K2_statistics(stream_s0_to_k2, stream_stat);
    K3_normalize(stream_s1, stream_stat, stream_s3);
    K4_and_store(stream_s3, out);
}
