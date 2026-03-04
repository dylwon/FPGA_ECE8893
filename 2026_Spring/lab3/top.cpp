#include "dcl.h"
#include <hls_stream.h>
#include <ap_int.h>

#define TILE_FACTOR 32
#define VEC_N (N / TILE_FACTOR)
#define VEC_BLOCK (BLOCK / TILE_FACTOR)

// The new Dual-Fetch data type
typedef ap_uint<1024> uint1024_dt;

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
// 1. PURE AXI READ (Decoupled from Math)
// --------------------------------------------------------
void read_in(const data_t in[N], hls::stream<uint1024_dt>& raw_stream) {
    #pragma HLS INLINE off
    const uint1024_dt* in_ptr = (const uint1024_dt*)in;
    
    read_loop: for (int i = 0; i < VEC_N; i++) {
        #pragma HLS PIPELINE II=1
        raw_stream.write(in_ptr[i]); 
    }
}

// --------------------------------------------------------
// 2. K0: Preprocess and Split Streams (Pipelined Math)
// --------------------------------------------------------
void K0_preprocess(hls::stream<uint1024_dt>& raw_stream, hls::stream<uint1024_dt>& out_k1, hls::stream<uint1024_dt>& out_k2) {
    #pragma HLS INLINE off
    const coef_t alpha = (coef_t)0.875;
    const coef_t beta  = (coef_t)0.125;

    k0_loop: for (int i = 0; i < VEC_N; i++) {
        #pragma HLS PIPELINE II=1
        uint1024_dt block = raw_stream.read(); 
        uint1024_dt out_block = 0;

        for (int k = 0; k < TILE_FACTOR; k++) {
            #pragma HLS UNROLL
            ap_uint<32> tmp_in = block(k * 32 + 31, k * 32);
            data_t val;
            val.range() = tmp_in; 
            
            // Pipelined Math to save clock period slack
            acc_t mul_alpha = (acc_t)alpha * (acc_t)val;
            #pragma HLS BIND_OP variable=mul_alpha op=mul impl=dsp latency=2
            
            data_t s0_val = (data_t)(mul_alpha + (acc_t)beta);
            #pragma HLS BIND_OP variable=s0_val op=add impl=fabric latency=1
            
            out_block(k * 32 + 31, k * 32) = s0_val.range(); 
        }
        out_k1.write(out_block);
        out_k2.write(out_block);
    }
}

// --------------------------------------------------------
// K1: Transform Branchless Sliding Window (Bit-Shift Optimized)
// --------------------------------------------------------
void K1_transform(hls::stream<uint1024_dt>& in_stream, hls::stream<uint1024_dt>& out_stream) {
    #pragma HLS INLINE off
    const coef_t w0 = (coef_t)0.50;
    const coef_t w1 = (coef_t)(-0.25);
    const coef_t w2 = (coef_t)0.125;

    data_t delay_reg[2] = {0, 0};
    #pragma HLS ARRAY_PARTITION variable=delay_reg complete

    k1_loop: for (int i = 0; i < VEC_N; i++) {
        #pragma HLS PIPELINE II=1
        uint1024_dt block = in_stream.read();
        uint1024_dt out_block = 0;

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

            // 1. Let the compiler infer 0-delay bit-shifts for powers of 2!
            acc_t m0 = (acc_t)w0 * (acc_t)x0;
            acc_t m1 = (acc_t)w1 * (acc_t)x1;
            acc_t m2 = (acc_t)w2 * (acc_t)x2;

            acc_t sum1 = m0 + m1;
            #pragma HLS BIND_OP variable=sum1 op=add impl=fabric latency=1
            
            acc_t acc  = sum1 + m2;
            #pragma HLS BIND_OP variable=acc op=add impl=fabric latency=1
            
            // Force a hard 2-cycle routing break before the heavy clamp logic
            data_t y;
            {
                #pragma HLS LATENCY min=2 max=2
                data_t abs_val = abs_fp((data_t)acc);
                y = clamp_fp(abs_val, (data_t)0, (data_t)7.5);
            }
            out_block(k * 32 + 31, k * 32) = y.range();
        }
        out_stream.write(out_block);
    }
}

// --------------------------------------------------------
// K2: Per-Block Statistic (Clean C++ with Pipelined Tree)
// --------------------------------------------------------
void K2_statistics(hls::stream<uint1024_dt>& in_stream, hls::stream<stat_t>& stat_stream) {
    #pragma HLS INLINE off
    const stat_t eps = (stat_t)0.5;
    acc_t sum_abs = 0;

    k2_loop: for (int i = 0; i < VEC_N; i++) {
        #pragma HLS PIPELINE II=1
        uint1024_dt block = in_stream.read();
        
        acc_t local_sum = 0;

        for (int k = 0; k < TILE_FACTOR; k++) {
            #pragma HLS UNROLL
            ap_uint<32> tmp = block(k * 32 + 31, k * 32);
            data_t val;
            val.range() = tmp;
            local_sum += (acc_t)abs_fp(val);
        }
        
        // Force 4 pipeline registers inside the 32-input tree!
        #pragma HLS BIND_OP variable=local_sum op=add impl=fabric latency=4
        
        sum_abs += local_sum;

        if ((i + 1) % VEC_BLOCK == 0) {
            stat_t avg_abs = (stat_t)(sum_abs / (acc_t)BLOCK);
            stat_stream.write(avg_abs + eps);
            sum_abs = 0; 
        }
    }
}

// --------------------------------------------------------
// 5. K3A: Isolated Hardware Divider
// --------------------------------------------------------
void K3A_divide(hls::stream<stat_t>& stat_stream, hls::stream<stat_t>& inv_stat_stream) {
    #pragma HLS INLINE off
    
    divide_loop: for (int b = 0; b < (N / BLOCK); b++) {
        #pragma HLS PIPELINE II=1
        
        stat_t st = stat_stream.read();
        
        // Let Vitis HLS auto-infer the best pipelined architecture for this division
        stat_t inv_st = (stat_t)((acc_t)1 / (acc_t)st);
        
        inv_stat_stream.write(inv_st);
    }
}

// --------------------------------------------------------
// 6. K3B: Join and Normalize (Pipelined Math)
// --------------------------------------------------------
void K3B_normalize(hls::stream<uint1024_dt>& in_stream, hls::stream<stat_t>& inv_stat_stream, hls::stream<uint1024_dt>& out_stream) {
    #pragma HLS INLINE off
    stat_t inv_st = 0;

    k3b_loop: for (int i = 0; i < VEC_N; i++) {
        #pragma HLS PIPELINE II=1
        
        if (i % VEC_BLOCK == 0) {
            inv_st = inv_stat_stream.read();
        }

        uint1024_dt block = in_stream.read();
        uint1024_dt out_block = 0;

        for (int k = 0; k < TILE_FACTOR; k++) {
            #pragma HLS UNROLL
            ap_uint<32> tmp = block(k * 32 + 31, k * 32);
            data_t val;
            val.range() = tmp;

            // Pipelined multiplier
            data_t norm_val = (data_t)((acc_t)val * (acc_t)inv_st);
            #pragma HLS BIND_OP variable=norm_val op=mul impl=dsp latency=2

            out_block(k * 32 + 31, k * 32) = norm_val.range();
        }
        out_stream.write(out_block);
    }
}

// --------------------------------------------------------
// 7. K4: Postprocess (Pipelined Math)
// --------------------------------------------------------
void K4_postprocess(hls::stream<uint1024_dt>& in_stream, hls::stream<uint1024_dt>& out_stream) {
    #pragma HLS INLINE off
    const coef_t gamma = (coef_t)1.25;
    const coef_t delta = (coef_t)0.05;

    k4_loop: for (int i = 0; i < VEC_N; i++) {
        #pragma HLS PIPELINE II=1
        uint1024_dt block = in_stream.read();
        uint1024_dt write_block = 0;

        for (int k = 0; k < TILE_FACTOR; k++) {
            #pragma HLS UNROLL
            ap_uint<32> tmp = block(k * 32 + 31, k * 32);
            data_t val;
            val.range() = tmp;
            
            // Pipelined Math
            acc_t mul_gamma = (acc_t)gamma * (acc_t)val;
            #pragma HLS BIND_OP variable=mul_gamma op=mul impl=dsp latency=2
            
            data_t z = (data_t)(mul_gamma + (acc_t)delta);
            #pragma HLS BIND_OP variable=z op=add impl=fabric latency=1

            data_t clamped = clamp_fp(z, (data_t)0, (data_t)7.9);
            write_block(k * 32 + 31, k * 32) = clamped.range();
        }
        out_stream.write(write_block); 
    }
}

// --------------------------------------------------------
// 8. PURE AXI WRITE (Decoupled from Math)
// --------------------------------------------------------
void write_out(hls::stream<uint1024_dt>& final_stream, data_t out[N]) {
    #pragma HLS INLINE off
    uint1024_dt* out_ptr = (uint1024_dt*)out;
    
    write_loop: for (int i = 0; i < VEC_N; i++) {
        #pragma HLS PIPELINE II=1
        out_ptr[i] = final_stream.read(); 
    }
}

// --------------------------------------------------------
// TOP KERNEL
// --------------------------------------------------------
void top_kernel(const data_t in[N], data_t out[N]) {
    #pragma HLS interface m_axi port=in offset=slave bundle=gmem0 max_read_burst_length=256 num_read_outstanding=4 max_widen_bitwidth=1024
    #pragma HLS interface m_axi port=out offset=slave bundle=gmem1 max_write_burst_length=256 num_write_outstanding=4 max_widen_bitwidth=1024
    #pragma HLS interface s_axilite port=return

    // New decoupled AXI streams
    hls::stream<uint1024_dt> stream_raw("stream_raw");
    #pragma HLS STREAM variable=stream_raw depth=8

    hls::stream<uint1024_dt> stream_final("stream_final");
    #pragma HLS STREAM variable=stream_final depth=8

    hls::stream<uint1024_dt> stream_s0_to_k1("stream_s0_to_k1");
    #pragma HLS STREAM variable=stream_s0_to_k1 depth=4
    #pragma HLS BIND_STORAGE variable=stream_s0_to_k1 type=fifo impl=srl

    hls::stream<uint1024_dt> stream_s0_to_k2("stream_s0_to_k2");
    #pragma HLS STREAM variable=stream_s0_to_k2 depth=16

    hls::stream<uint1024_dt> stream_s1("stream_s1");
    #pragma HLS STREAM variable=stream_s1 depth=4096
    #pragma HLS BIND_STORAGE variable=stream_s1 type=fifo impl=bram

    hls::stream<stat_t> stream_stat("stream_stat");
    #pragma HLS STREAM variable=stream_stat depth=16

    hls::stream<stat_t> stream_inv_stat("stream_inv_stat");
    #pragma HLS STREAM variable=stream_inv_stat depth=16

    hls::stream<uint1024_dt> stream_s3("stream_s3");
    #pragma HLS STREAM variable=stream_s3 depth=16

    #pragma HLS DATAFLOW

    read_in(in, stream_raw);
    K0_preprocess(stream_raw, stream_s0_to_k1, stream_s0_to_k2);
    K1_transform(stream_s0_to_k1, stream_s1);
    K2_statistics(stream_s0_to_k2, stream_stat);
    
    // Separated Divider
    K3A_divide(stream_stat, stream_inv_stat);
    K3B_normalize(stream_s1, stream_inv_stat, stream_s3);
    
    K4_postprocess(stream_s3, stream_final);
    write_out(stream_final, out);
}
