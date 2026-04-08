#include "dcl.h"
#include <hls_stream.h>

typedef ap_uint<512> super_wide_t; 

struct pixel4_t {
    pixel_t p0, p1, p2, p3; 
};

// ---------------------------------------------------------
// 512-Bit AXI Burst Reader
// ---------------------------------------------------------
void read_input_burst(
    pixel_t img_r[HEIGHT][WIDTH], 
    pixel_t img_g[HEIGHT][WIDTH], 
    pixel_t img_b[HEIGHT][WIDTH], 
    hls::stream<pixel4_t>& stream_r,
    hls::stream<pixel4_t>& stream_g,
    hls::stream<pixel4_t>& stream_b
) {
    super_wide_t *flat_r = (super_wide_t*)img_r;
    super_wide_t *flat_g = (super_wide_t*)img_g;
    super_wide_t *flat_b = (super_wide_t*)img_b;

    super_wide_t chunk_r, chunk_g, chunk_b;

    for (int i = 0; i < (HEIGHT * WIDTH) / 4; i++) {
        #pragma HLS PIPELINE II=1
        
        // Fetch a new 512-bit line from DRAM every 8 iterations (8 * 4 = 32 pixels)
        if (i % 8 == 0) {
            chunk_r = flat_r[i / 8];
            chunk_g = flat_g[i / 8];
            chunk_b = flat_b[i / 8];
        }
        
        int offset = (i % 8) * 64;
        pixel4_t vec_r, vec_g, vec_b;
        
        vec_r.p0.range() = chunk_r.range(offset + 15,  offset + 0);
        vec_r.p1.range() = chunk_r.range(offset + 31,  offset + 16);
        vec_r.p2.range() = chunk_r.range(offset + 47,  offset + 32);
        vec_r.p3.range() = chunk_r.range(offset + 63,  offset + 48);

        vec_g.p0.range() = chunk_g.range(offset + 15,  offset + 0);
        vec_g.p1.range() = chunk_g.range(offset + 31,  offset + 16);
        vec_g.p2.range() = chunk_g.range(offset + 47,  offset + 32);
        vec_g.p3.range() = chunk_g.range(offset + 63,  offset + 48);

        vec_b.p0.range() = chunk_b.range(offset + 15,  offset + 0);
        vec_b.p1.range() = chunk_b.range(offset + 31,  offset + 16);
        vec_b.p2.range() = chunk_b.range(offset + 47,  offset + 32);
        vec_b.p3.range() = chunk_b.range(offset + 63,  offset + 48);

        stream_r.write(vec_r);
        stream_g.write(vec_g);
        stream_b.write(vec_b);
    }
}


// ---------------------------------------------------------
// BT.601 Conversion
// ---------------------------------------------------------
inline pixel_t bt601_convert(pixel_t r_pix, pixel_t g_pix, pixel_t b_pix) {
    #pragma HLS INLINE
    int red = (int)r_pix; 
    int green = (int)g_pix; 
    int blue = (int)b_pix;
    int y_calc = (66 * red + 129 * green + 25 * blue + 4096) >> 8;

    if (y_calc < 16) {
        y_calc = 16;
    }
    else if (y_calc > 235) {
        y_calc = 235;
    }

    int normalized_gray = ((y_calc - 16) * 298) >> 8;

    if (normalized_gray < 0) {
        normalized_gray = 0;
    }
    else if (normalized_gray > 255) {
        normalized_gray = 255;
    }

    return (pixel_t)normalized_gray;
}

// ---------------------------------------------------------
// Kernel 1: YCbCr Converter
// ---------------------------------------------------------
void kernel1_rgb_to_ycbcr_simd(
    hls::stream<pixel4_t>& stream_r, 
    hls::stream<pixel4_t>& stream_g, 
    hls::stream<pixel4_t>& stream_b, 
    hls::stream<pixel4_t>& stream_gray
) {
    for (int i = 0; i < (HEIGHT * WIDTH) / 4; i++) {
        #pragma HLS PIPELINE II=1

        pixel4_t in_r = stream_r.read();
        pixel4_t in_g = stream_g.read();
        pixel4_t in_b = stream_b.read();
        pixel4_t out_gray;

        out_gray.p0 = bt601_convert(in_r.p0, in_g.p0, in_b.p0);
        out_gray.p1 = bt601_convert(in_r.p1, in_g.p1, in_b.p1);
        out_gray.p2 = bt601_convert(in_r.p2, in_g.p2, in_b.p2);
        out_gray.p3 = bt601_convert(in_r.p3, in_g.p3, in_b.p3);

        stream_gray.write(out_gray);
    }
}

// ---------------------------------------------------------
// Kernel 2: Odd-Even Median Filter
// ---------------------------------------------------------
void kernel2_median_simd(hls::stream<pixel4_t>& stream_in, hls::stream<pixel4_t>& stream_out) {
    static pixel4_t line_buf[4][WIDTH / 4];
    #pragma HLS bind_storage variable=line_buf type=ram_t2p impl=bram
    #pragma HLS array_partition variable=line_buf complete dim=1
    
    pixel_t window[5][12];
    #pragma HLS ARRAY_PARTITION variable=window complete dim=0

    for (int r = 0; r < HEIGHT + 2; r++) {
        for (int c = 0; c < (WIDTH / 4) + 1; c++) {
            #pragma HLS PIPELINE II=1

            pixel4_t new_vec;
            new_vec.p0 = 0; new_vec.p1 = 0; new_vec.p2 = 0; new_vec.p3 = 0;

            if (r < HEIGHT && c < WIDTH / 4) new_vec = stream_in.read();

            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 8; j++) window[i][j] = window[i][j+4];
            }

            int c_idx = (c < WIDTH / 4) ? c : (WIDTH / 4) - 1;
            pixel4_t col_val0 = line_buf[0][c_idx]; pixel4_t col_val1 = line_buf[1][c_idx];
            pixel4_t col_val2 = line_buf[2][c_idx]; pixel4_t col_val3 = line_buf[3][c_idx];

            window[0][8] = col_val0.p0; window[0][9] = col_val0.p1; window[0][10] = col_val0.p2; window[0][11] = col_val0.p3;
            window[1][8] = col_val1.p0; window[1][9] = col_val1.p1; window[1][10] = col_val1.p2; window[1][11] = col_val1.p3;
            window[2][8] = col_val2.p0; window[2][9] = col_val2.p1; window[2][10] = col_val2.p2; window[2][11] = col_val2.p3;
            window[3][8] = col_val3.p0; window[3][9] = col_val3.p1; window[3][10] = col_val3.p2; window[3][11] = col_val3.p3;
            window[4][8] = new_vec.p0;  window[4][9] = new_vec.p1;  window[4][10] = new_vec.p2;  window[4][11] = new_vec.p3;

            if (c < WIDTH / 4) {
                line_buf[0][c] = col_val1; line_buf[1][c] = col_val2;
                line_buf[2][c] = col_val3; line_buf[3][c] = new_vec;
            }

            if (r >= 2 && c >= 1) { 
                pixel4_t out_vec;
                pixel_t out_arr[4];
                #pragma HLS ARRAY_PARTITION variable=out_arr complete

                for (int p = 0; p < 4; p++) {
                    #pragma HLS UNROLL
                    int out_r = r - 2; int out_c = (c - 1) * 4 + p; 
                    
                    if (out_r < 2 || out_r >= HEIGHT - 2 || out_c < 2 || out_c >= WIDTH - 2) {
                        out_arr[p] = window[2][p + 4]; 
                    } else {
                        // Flatten the 5x5 window into a 25-element array
                        pixel_t flat[25];
                        #pragma HLS ARRAY_PARTITION variable=flat complete
                        int idx = 0;
                        for (int kr = 0; kr < 5; kr++) {
                            for (int kc = 0; kc < 5; kc++) {
                                flat[idx++] = window[kr][p + 2 + kc];
                            }
                        }

                        // Odd-Even Bubble Sort
                        for (int i = 0; i < 25; i++) {
                            #pragma HLS UNROLL
                            if (i % 2 == 0) { // Even phase
                                for (int j = 0; j < 24; j += 2) {
                                    #pragma HLS UNROLL
                                    pixel_t a = flat[j]; pixel_t b = flat[j+1];
                                    
                                    bool is_greater_raw = (a > b);
                                    bool is_greater;

                                    #pragma HLS bind_op variable=is_greater op=add impl=fabric latency=1
                                    is_greater = is_greater_raw + 0;
                                    
                                    flat[j]   = is_greater ? b : a;
                                    flat[j+1] = is_greater ? a : b;
                                }
                            } else { // Odd phase
                                for (int j = 1; j < 24; j += 2) {
                                    #pragma HLS UNROLL
                                    pixel_t a = flat[j]; pixel_t b = flat[j+1];
                                    
                                    bool is_greater_raw = (a > b);
                                    bool is_greater;

                                    #pragma HLS bind_op variable=is_greater op=add impl=fabric latency=1
                                    is_greater = is_greater_raw + 0;
                                    
                                    flat[j]   = is_greater ? b : a;
                                    flat[j+1] = is_greater ? a : b;
                                }
                            }
                        }
                        // Extract median
                        out_arr[p] = flat[12];
                    }
                }
                out_vec.p0 = out_arr[0]; out_vec.p1 = out_arr[1]; out_vec.p2 = out_arr[2]; out_vec.p3 = out_arr[3];
                stream_out.write(out_vec);
            }
        }
    }
}

// ---------------------------------------------------------
// Kernel 3: Bilateral Filter
// ---------------------------------------------------------
void kernel3_bilateral_simd(hls::stream<pixel4_t>& stream_in, hls::stream<pixel4_t>& stream_out) {
    
    const int spatial_w[5][5] = {
        { 1,  4,  7,  4,  1}, { 4, 16, 26, 16,  4}, { 7, 26, 41, 26,  7},
        { 4, 16, 26, 16,  4}, { 1,  4,  7,  4,  1}
    };

    const int color_w_lut[32] = {
        255, 245, 220, 183, 141, 101, 67, 41, 23, 12, 6, 3, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    static pixel4_t line_buf[4][WIDTH / 4];
    #pragma HLS bind_storage variable=line_buf type=ram_t2p impl=bram
    #pragma HLS array_partition variable=line_buf complete dim=1
    
    pixel_t window[5][12];
    #pragma HLS ARRAY_PARTITION variable=window complete dim=0

    for (int r = 0; r < HEIGHT + 2; r++) {
        for (int c = 0; c < (WIDTH / 4) + 1; c++) {
            #pragma HLS PIPELINE II=1

            pixel4_t new_vec;
            new_vec.p0 = 0; new_vec.p1 = 0; new_vec.p2 = 0; new_vec.p3 = 0;

            if (r < HEIGHT && c < WIDTH / 4) new_vec = stream_in.read();

            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 8; j++) {
                    window[i][j] = window[i][j+4];
                }
            }

            int c_idx = (c < WIDTH / 4) ? c : (WIDTH / 4) - 1;
            pixel4_t col_val0 = line_buf[0][c_idx];
            pixel4_t col_val1 = line_buf[1][c_idx];
            pixel4_t col_val2 = line_buf[2][c_idx];
            pixel4_t col_val3 = line_buf[3][c_idx];

            window[0][8] = col_val0.p0; window[0][9] = col_val0.p1; window[0][10] = col_val0.p2; window[0][11] = col_val0.p3;
            window[1][8] = col_val1.p0; window[1][9] = col_val1.p1; window[1][10] = col_val1.p2; window[1][11] = col_val1.p3;
            window[2][8] = col_val2.p0; window[2][9] = col_val2.p1; window[2][10] = col_val2.p2; window[2][11] = col_val2.p3;
            window[3][8] = col_val3.p0; window[3][9] = col_val3.p1; window[3][10] = col_val3.p2; window[3][11] = col_val3.p3;
            window[4][8] = new_vec.p0;  window[4][9] = new_vec.p1;  window[4][10] = new_vec.p2;  window[4][11] = new_vec.p3;

            if (c < WIDTH / 4) {
                line_buf[0][c] = col_val1;
                line_buf[1][c] = col_val2;
                line_buf[2][c] = col_val3;
                line_buf[3][c] = new_vec;
            }

            if (r >= 2 && c >= 1) { 
                pixel4_t out_vec;
                pixel_t out_arr[4];
                #pragma HLS ARRAY_PARTITION variable=out_arr complete

                for (int p = 0; p < 4; p++) {
                    #pragma HLS UNROLL
                    
                    int out_r = r - 2; 
                    int out_c = (c - 1) * 4 + p; 
                    
                    if (out_r < 2 || out_r >= HEIGHT - 2 || out_c < 2 || out_c >= WIDTH - 2) {
                        out_arr[p] = window[2][p + 4]; 
                    } else {
                        int center_val = (int)window[2][p + 4];
                        
                        int val_row_sum[5] = {0, 0, 0, 0, 0};
                        int weight_row_sum[5] = {0, 0, 0, 0, 0};
                        
                        #pragma HLS ARRAY_PARTITION variable=val_row_sum complete
                        #pragma HLS ARRAY_PARTITION variable=weight_row_sum complete

                        // Parallel Multiplier & Row-Sum Tree
                        for (int kr = 0; kr < 5; kr++) {
                            #pragma HLS UNROLL
                            for (int kc = 0; kc < 5; kc++) {
                                #pragma HLS UNROLL
                                int neighbor_val = (int)window[kr][p + 2 + kc];
                                
                                int raw_diff = center_val - neighbor_val;
                                int diff = (raw_diff < 0 ? -raw_diff : raw_diff) >> 3;
                                if (diff > 31) diff = 31;
                                
                                int w = spatial_w[kr][kc] * color_w_lut[diff];
                                
                                int mult_val = neighbor_val * w;
                                #pragma HLS bind_op variable=mult_val op=mul impl=dsp latency=3
                                
                                // Accumulate only within the specific row
                                val_row_sum[kr] += mult_val;
                                weight_row_sum[kr] += w;
                            }
                        }
                        
                        int val_sum = 0;
                        int weight_sum = 0;
                        
                        #pragma HLS bind_op variable=val_sum op=add impl=fabric latency=1
                        #pragma HLS bind_op variable=weight_sum op=add impl=fabric latency=1
                        
                        val_sum = val_row_sum[0] + val_row_sum[1] + val_row_sum[2] + val_row_sum[3] + val_row_sum[4];
                        weight_sum = weight_row_sum[0] + weight_row_sum[1] + weight_row_sum[2] + weight_row_sum[3] + weight_row_sum[4];

                        int div_result;
                        #pragma HLS bind_op variable=div_result op=sdiv impl=auto
                        div_result = val_sum / weight_sum;

                        out_arr[p] = (pixel_t)div_result;
                    }
                }
                
                out_vec.p0 = out_arr[0]; out_vec.p1 = out_arr[1]; out_vec.p2 = out_arr[2]; out_vec.p3 = out_arr[3];
                stream_out.write(out_vec);
            }
        }
    }
}

// ---------------------------------------------------------
// Kernel 4: Sobel Gradients
// ---------------------------------------------------------
void kernel4_sobel_simd(hls::stream<pixel4_t>& stream_in, hls::stream<pixel4_t>& stream_x, hls::stream<pixel4_t>& stream_y) {
    coef_t sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    coef_t sobel_y[3][3] = {{ 1, 2, 1}, { 0, 0, 0}, {-1,-2,-1}};
    #pragma HLS ARRAY_PARTITION variable=sobel_x complete dim=0
    #pragma HLS ARRAY_PARTITION variable=sobel_y complete dim=0

    static pixel4_t line_buf[2][WIDTH / 4];
    #pragma HLS bind_storage variable=line_buf type=ram_t2p impl=bram
    #pragma HLS array_partition variable=line_buf complete dim=1
    
    pixel_t window[3][12];
    #pragma HLS ARRAY_PARTITION variable=window complete dim=0

    for (int r = 0; r < HEIGHT + 1; r++) {
        for (int c = 0; c < (WIDTH / 4) + 1; c++) {
            #pragma HLS PIPELINE II=1

            pixel4_t new_vec;
            new_vec.p0 = 0; new_vec.p1 = 0; new_vec.p2 = 0; new_vec.p3 = 0;
            if (r < HEIGHT && c < WIDTH / 4) new_vec = stream_in.read();

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 8; j++) {
                    window[i][j] = window[i][j+4];
                }
            }

            int c_idx = (c < WIDTH / 4) ? c : (WIDTH / 4) - 1;
            pixel4_t col_val0 = line_buf[0][c_idx];
            pixel4_t col_val1 = line_buf[1][c_idx];

            window[0][8] = col_val0.p0; window[0][9] = col_val0.p1; window[0][10] = col_val0.p2; window[0][11] = col_val0.p3;
            window[1][8] = col_val1.p0; window[1][9] = col_val1.p1; window[1][10] = col_val1.p2; window[1][11] = col_val1.p3;
            window[2][8] = new_vec.p0;  window[2][9] = new_vec.p1;  window[2][10] = new_vec.p2;  window[2][11] = new_vec.p3;

            if (c < WIDTH / 4) {
                line_buf[0][c] = col_val1;
                line_buf[1][c] = new_vec;
            }

            if (r >= 1 && c >= 1) { 
                pixel4_t out_x, out_y;
                pixel_t out_x_arr[4], out_y_arr[4];
                #pragma HLS ARRAY_PARTITION variable=out_x_arr complete
                #pragma HLS ARRAY_PARTITION variable=out_y_arr complete

                for (int p = 0; p < 4; p++) {
                    #pragma HLS UNROLL
                    int out_r = r - 1; int out_c = (c - 1) * 4 + p; 
                    
                    if (out_r == 0 || out_r == HEIGHT - 1 || out_c == 0 || out_c == WIDTH - 1) {
                        out_x_arr[p] = 0; out_y_arr[p] = 0;
                    } else {
                        calc_t sum_x = 0; calc_t sum_y = 0;
                        for (int kr = 0; kr < 3; kr++) {
                            for (int kc = 0; kc < 3; kc++) {
                                sum_x += window[kr][p + 3 + kc] * sobel_x[kr][kc];
                                sum_y += window[kr][p + 3 + kc] * sobel_y[kr][kc];
                            }
                        }
                        out_x_arr[p] = (pixel_t)sum_x; 
                        out_y_arr[p] = (pixel_t)sum_y;
                    }
                }
                out_x.p0 = out_x_arr[0]; out_x.p1 = out_x_arr[1]; out_x.p2 = out_x_arr[2]; out_x.p3 = out_x_arr[3];
                out_y.p0 = out_y_arr[0]; out_y.p1 = out_y_arr[1]; out_y.p2 = out_y_arr[2]; out_y.p3 = out_y_arr[3];
                stream_x.write(out_x);
                stream_y.write(out_y);
            }
        }
    }
}

// ---------------------------------------------------------
// Kernel 5: Magnitude & Direction
// ---------------------------------------------------------
void kernel5_mag_dir_simd(
    hls::stream<pixel4_t>& stream_x, hls::stream<pixel4_t>& stream_y, 
    hls::stream<pixel4_t>& stream_mag, hls::stream<pixel4_t>& stream_dir
) {
    for (int i = 0; i < (HEIGHT * WIDTH) / 4; i++) {
        #pragma HLS PIPELINE II=1

        pixel4_t in_x = stream_x.read();
        pixel4_t in_y = stream_y.read();

        pixel4_t out_mag, out_dir;
        
        calc_t gx[4] = {(calc_t)in_x.p0, (calc_t)in_x.p1, (calc_t)in_x.p2, (calc_t)in_x.p3};
        calc_t gy[4] = {(calc_t)in_y.p0, (calc_t)in_y.p1, (calc_t)in_y.p2, (calc_t)in_y.p3};
        
        pixel_t mag_arr[4];
        pixel_t dir_arr[4];

        for (int p = 0; p < 4; p++) {
            #pragma HLS UNROLL
            
            calc_t curr_gx = gx[p];
            calc_t curr_gy = gy[p];
            
            calc_t abs_gx = (curr_gx < 0) ? (calc_t)(-curr_gx) : curr_gx;
            calc_t abs_gy = (curr_gy < 0) ? (calc_t)(-curr_gy) : curr_gy;
            mag_arr[p] = (pixel_t)(abs_gx + abs_gy);

            pixel_t dir = 0;
            if (curr_gx == 0) {
                dir = 90; 
            } else {
                calc_t slope;
                #pragma HLS bind_op variable=slope op=sdiv impl=auto
                slope = curr_gy / curr_gx; 

                if (slope > (calc_t)-0.414 && slope <= (calc_t)0.414) dir = 0;
                else if (slope > (calc_t)0.414 && slope <= (calc_t)2.414) dir = 45;
                else if (slope < (calc_t)-0.414 && slope >= (calc_t)-2.414) dir = 135;
                else dir = 90;
            }
            dir_arr[p] = dir;
        }

        out_mag.p0 = mag_arr[0]; out_mag.p1 = mag_arr[1]; out_mag.p2 = mag_arr[2]; out_mag.p3 = mag_arr[3];
        out_dir.p0 = dir_arr[0]; out_dir.p1 = dir_arr[1]; out_dir.p2 = dir_arr[2]; out_dir.p3 = dir_arr[3];

        stream_mag.write(out_mag);
        stream_dir.write(out_dir);
    }
}

// ---------------------------------------------------------
// Kernel 6: Non-Maximum Suppression
// ---------------------------------------------------------
void kernel6_nms_simd(hls::stream<pixel4_t>& stream_mag, hls::stream<pixel4_t>& stream_dir, hls::stream<pixel4_t>& stream_out) {
    static pixel4_t mag_buf[2][WIDTH / 4];
    static pixel4_t dir_buf[2][WIDTH / 4];
    #pragma HLS bind_storage variable=mag_buf type=ram_t2p impl=bram
    #pragma HLS bind_storage variable=dir_buf type=ram_t2p impl=bram
    #pragma HLS ARRAY_PARTITION variable=mag_buf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=dir_buf complete dim=1

    pixel_t mag_win[3][12], dir_win[3][12];
    #pragma HLS ARRAY_PARTITION variable=mag_win complete dim=0
    #pragma HLS ARRAY_PARTITION variable=dir_win complete dim=0

    for (int r = 0; r < HEIGHT + 1; r++) {
        for (int c = 0; c < (WIDTH / 4) + 1; c++) {
            #pragma HLS PIPELINE II=1

            pixel4_t new_mag, new_dir;
            new_mag.p0 = 0; new_mag.p1 = 0; new_mag.p2 = 0; new_mag.p3 = 0;
            new_dir.p0 = 0; new_dir.p1 = 0; new_dir.p2 = 0; new_dir.p3 = 0;

            if (r < HEIGHT && c < WIDTH / 4) {
                new_mag = stream_mag.read(); 
                new_dir = stream_dir.read();
            }

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 8; j++) {
                    mag_win[i][j] = mag_win[i][j+4];
                    dir_win[i][j] = dir_win[i][j+4];
                }
            }

            int c_idx = (c < WIDTH / 4) ? c : (WIDTH / 4) - 1;
            pixel4_t col_m0 = mag_buf[0][c_idx]; pixel4_t col_m1 = mag_buf[1][c_idx];
            pixel4_t col_d0 = dir_buf[0][c_idx]; pixel4_t col_d1 = dir_buf[1][c_idx];

            mag_win[0][8] = col_m0.p0; mag_win[0][9] = col_m0.p1; mag_win[0][10] = col_m0.p2; mag_win[0][11] = col_m0.p3;
            mag_win[1][8] = col_m1.p0; mag_win[1][9] = col_m1.p1; mag_win[1][10] = col_m1.p2; mag_win[1][11] = col_m1.p3;
            mag_win[2][8] = new_mag.p0; mag_win[2][9] = new_mag.p1; mag_win[2][10] = new_mag.p2; mag_win[2][11] = new_mag.p3;

            dir_win[0][8] = col_d0.p0; dir_win[0][9] = col_d0.p1; dir_win[0][10] = col_d0.p2; dir_win[0][11] = col_d0.p3;
            dir_win[1][8] = col_d1.p0; dir_win[1][9] = col_d1.p1; dir_win[1][10] = col_d1.p2; dir_win[1][11] = col_d1.p3;
            dir_win[2][8] = new_dir.p0; dir_win[2][9] = new_dir.p1; dir_win[2][10] = new_dir.p2; dir_win[2][11] = new_dir.p3;

            if (c < WIDTH / 4) {
                mag_buf[0][c] = col_m1; mag_buf[1][c] = new_mag;
                dir_buf[0][c] = col_d1; dir_buf[1][c] = new_dir;
            }

            if (r >= 1 && c >= 1) {
                pixel4_t out_vec;
                pixel_t out_arr[4];
                #pragma HLS ARRAY_PARTITION variable=out_arr complete

                for (int p = 0; p < 4; p++) {
                    #pragma HLS UNROLL
                    int out_r = r - 1; int out_c = (c - 1) * 4 + p;
                    if (out_r == 0 || out_r == HEIGHT - 1 || out_c == 0 || out_c == WIDTH - 1) {
                        out_arr[p] = 0;
                    } else {
                        pixel_t mag = mag_win[1][p + 4];
                        pixel_t dir = dir_win[1][p + 4];
                        pixel_t m1 = 0, m2 = 0;
                        
                        if (dir == 0)      { m1 = mag_win[1][p + 3]; m2 = mag_win[1][p + 5]; }
                        else if (dir == 90){ m1 = mag_win[0][p + 4]; m2 = mag_win[2][p + 4]; }
                        else if (dir == 45){ m1 = mag_win[2][p + 3]; m2 = mag_win[0][p + 5]; }
                        else               { m1 = mag_win[0][p + 3]; m2 = mag_win[2][p + 5]; }

                        out_arr[p] = (mag >= m1 && mag >= m2) ? mag : (pixel_t)0;
                    }
                }
                out_vec.p0 = out_arr[0]; out_vec.p1 = out_arr[1]; out_vec.p2 = out_arr[2]; out_vec.p3 = out_arr[3];
                stream_out.write(out_vec);
            }
        }
    }
}

// ---------------------------------------------------------
// Kernel 7: Adaptive Hysteresis
// ---------------------------------------------------------
void kernel7_hysteresis_simd(hls::stream<pixel4_t>& stream_in, hls::stream<pixel4_t>& stream_out) {
    static pixel4_t line_buf[4][WIDTH / 4];
    #pragma HLS bind_storage variable=line_buf type=ram_t2p impl=bram
    #pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1
    
    pixel_t window[5][12];
    #pragma HLS ARRAY_PARTITION variable=window complete dim=0

    for (int r = 0; r < HEIGHT + 2; r++) {
        for (int c = 0; c < (WIDTH / 4) + 1; c++) {
            #pragma HLS PIPELINE II=1

            pixel4_t new_vec;
            new_vec.p0 = 0; new_vec.p1 = 0; new_vec.p2 = 0; new_vec.p3 = 0;
            if (r < HEIGHT && c < WIDTH / 4) new_vec = stream_in.read();

            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 8; j++) window[i][j] = window[i][j+4];
            }

            int c_idx = (c < WIDTH / 4) ? c : (WIDTH / 4) - 1;
            pixel4_t col0 = line_buf[0][c_idx]; pixel4_t col1 = line_buf[1][c_idx];
            pixel4_t col2 = line_buf[2][c_idx]; pixel4_t col3 = line_buf[3][c_idx];

            window[0][8] = col0.p0; window[0][9] = col0.p1; window[0][10] = col0.p2; window[0][11] = col0.p3;
            window[1][8] = col1.p0; window[1][9] = col1.p1; window[1][10] = col1.p2; window[1][11] = col1.p3;
            window[2][8] = col2.p0; window[2][9] = col2.p1; window[2][10] = col2.p2; window[2][11] = col2.p3;
            window[3][8] = col3.p0; window[3][9] = col3.p1; window[3][10] = col3.p2; window[3][11] = col3.p3;
            window[4][8] = new_vec.p0; window[4][9] = new_vec.p1; window[4][10] = new_vec.p2; window[4][11] = new_vec.p3;

            if (c < WIDTH / 4) {
                line_buf[0][c] = col1; line_buf[1][c] = col2;
                line_buf[2][c] = col3; line_buf[3][c] = new_vec;
            }

            if (r >= 2 && c >= 1) { 
                pixel4_t out_vec;
                pixel_t out_arr[4];
                #pragma HLS ARRAY_PARTITION variable=out_arr complete

                for (int p = 0; p < 4; p++) {
                    #pragma HLS UNROLL
                    int out_r = r - 2; int out_c = (c - 1) * 4 + p; 
                    
                    if (out_r < 2 || out_r >= HEIGHT - 2 || out_c < 2 || out_c >= WIDTH - 2) {
                        out_arr[p] = 0;
                    } else {
                        calc_t row_sums[5] = {0, 0, 0, 0, 0};
                        #pragma HLS ARRAY_PARTITION variable=row_sums complete

                        for (int kr = 0; kr < 5; kr++) {
                            #pragma HLS UNROLL
                            for (int kc = 0; kc < 5; kc++) {
                                #pragma HLS UNROLL
                                row_sums[kr] += window[kr][p + 2 + kc];
                            }
                        }
                        
                        calc_t local_sum = 0;
                        #pragma HLS bind_op variable=local_sum op=add impl=fabric latency=1
                        
                        local_sum = row_sums[0] + row_sums[1] + row_sums[2] + row_sums[3] + row_sums[4];
                        
                        calc_t mean_div;
                        #pragma HLS bind_op variable=mean_div op=sdiv impl=auto
                        mean_div = local_sum / (calc_t)25;

                        pixel_t local_mean = (pixel_t)mean_div; 
                        pixel_t HIGH_THRESH = local_mean + (pixel_t)15;
                        pixel_t LOW_THRESH  = local_mean - (pixel_t)5;
                        pixel_t center_pixel = window[2][p + 4];
                        
                        if (center_pixel >= HIGH_THRESH) {
                            out_arr[p] = 255;
                        } else if (center_pixel < LOW_THRESH) {
                            out_arr[p] = 0;
                        } else {
                            bool connected = false;
                            for (int kr = 1; kr <= 3; kr++) {
                                #pragma HLS UNROLL
                                for (int kc = 1; kc <= 3; kc++) {
                                    #pragma HLS UNROLL
                                    if (kr == 2 && kc == 2) continue; 
                                    if (window[kr][p + 2 + kc] >= HIGH_THRESH) connected = true;
                                }
                            }
                            out_arr[p] = connected ? (pixel_t)255 : (pixel_t)0;
                        }
                    }
                }
                out_vec.p0 = out_arr[0]; out_vec.p1 = out_arr[1]; out_vec.p2 = out_arr[2]; out_vec.p3 = out_arr[3];
                stream_out.write(out_vec);
            }
        }
    }
}


// ---------------------------------------------------------
// Kernel 8: Morphological Dilation
// ---------------------------------------------------------
void kernel8_dilation_simd(hls::stream<pixel4_t>& stream_in, hls::stream<pixel4_t>& stream_out) {
    static pixel4_t line_buf[2][WIDTH / 4];
    #pragma HLS bind_storage variable=line_buf type=ram_t2p impl=bram
    #pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1
    
    pixel_t window[3][12];
    #pragma HLS ARRAY_PARTITION variable=window complete dim=0

    for (int r = 0; r < HEIGHT + 1; r++) {
        for (int c = 0; c < (WIDTH / 4) + 1; c++) {
            #pragma HLS PIPELINE II=1

            pixel4_t new_vec;
            new_vec.p0 = 0; new_vec.p1 = 0; new_vec.p2 = 0; new_vec.p3 = 0;
            if (r < HEIGHT && c < WIDTH / 4) new_vec = stream_in.read();

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 8; j++) {
                    window[i][j] = window[i][j+4];
                }
            }

            int c_idx = (c < WIDTH / 4) ? c : (WIDTH / 4) - 1;
            pixel4_t col0 = line_buf[0][c_idx]; pixel4_t col1 = line_buf[1][c_idx];

            window[0][8] = col0.p0; window[0][9] = col0.p1; window[0][10] = col0.p2; window[0][11] = col0.p3;
            window[1][8] = col1.p0; window[1][9] = col1.p1; window[1][10] = col1.p2; window[1][11] = col1.p3;
            window[2][8] = new_vec.p0; window[2][9] = new_vec.p1; window[2][10] = new_vec.p2; window[2][11] = new_vec.p3;

            if (c < WIDTH / 4) {
                line_buf[0][c] = col1; line_buf[1][c] = new_vec;
            }

            if (r >= 1 && c >= 1) {
                pixel4_t out_vec;
                pixel_t out_arr[4];
                #pragma HLS ARRAY_PARTITION variable=out_arr complete

                for (int p = 0; p < 4; p++) {
                    #pragma HLS UNROLL
                    int out_r = r - 1; int out_c = (c - 1) * 4 + p; 
                    
                    if (out_r == 0 || out_r == HEIGHT - 1 || out_c == 0 || out_c == WIDTH - 1) {
                        out_arr[p] = 0;
                    } else {
                        pixel_t max_val = 0;
                        for (int kr = 0; kr < 3; kr++) {
                            for (int kc = 0; kc < 3; kc++) {
                                if (window[kr][p + 3 + kc] == 255) max_val = 255;
                            }
                        }
                        out_arr[p] = max_val;
                    }
                }
                out_vec.p0 = out_arr[0]; out_vec.p1 = out_arr[1]; out_vec.p2 = out_arr[2]; out_vec.p3 = out_arr[3];
                stream_out.write(out_vec);
            }
        }
    }
}


// ---------------------------------------------------------
// Helper: 512-Bit AXI Burst Writer
// ---------------------------------------------------------
void write_output_burst(hls::stream<pixel4_t>& stream_in, pixel_t img_out[HEIGHT][WIDTH]) {
    super_wide_t *flat_out = (super_wide_t*)img_out;
    super_wide_t chunk = 0;
    
    for (int i = 0; i < (HEIGHT * WIDTH) / 4; i++) {
        #pragma HLS PIPELINE II=1
        pixel4_t vec = stream_in.read();
        
        chunk = chunk >> 64;
        
        chunk.range(511, 496) = vec.p3.range();
        chunk.range(495, 480) = vec.p2.range();
        chunk.range(479, 464) = vec.p1.range();
        chunk.range(463, 448) = vec.p0.range();
        
        if ((i % 8) == 7) {
            flat_out[i / 8] = chunk;
        }
    }
}

void top_kernel(
    pixel_t img_r[HEIGHT][WIDTH], 
    pixel_t img_g[HEIGHT][WIDTH], 
    pixel_t img_b[HEIGHT][WIDTH], 
    pixel_t img_out[HEIGHT][WIDTH]
) {
    #pragma HLS interface m_axi port=img_r offset=slave bundle=gmem0
    #pragma HLS interface m_axi port=img_g offset=slave bundle=gmem1
    #pragma HLS interface m_axi port=img_b offset=slave bundle=gmem2
    #pragma HLS interface m_axi port=img_out offset=slave bundle=gmem3
    #pragma HLS interface s_axilite port=return

    #pragma HLS DATAFLOW

    hls::stream<pixel4_t> stream_r_simd("r_simd");
    #pragma HLS STREAM variable=stream_r_simd depth=64
    #pragma HLS bind_storage variable=stream_r_simd type=fifo impl=srl

    hls::stream<pixel4_t> stream_g_simd("g_simd");
    #pragma HLS STREAM variable=stream_g_simd depth=64
    #pragma HLS bind_storage variable=stream_g_simd type=fifo impl=srl

    hls::stream<pixel4_t> stream_b_simd("b_simd");
    #pragma HLS STREAM variable=stream_b_simd depth=64
    #pragma HLS bind_storage variable=stream_b_simd type=fifo impl=srl

    hls::stream<pixel4_t> stream_gray_simd("gray_simd");
    #pragma HLS STREAM variable=stream_gray_simd depth=64
    #pragma HLS bind_storage variable=stream_gray_simd type=fifo impl=srl

    hls::stream<pixel4_t> stream_median_simd("median_simd");
    #pragma HLS STREAM variable=stream_median_simd depth=64
    #pragma HLS bind_storage variable=stream_median_simd type=fifo impl=srl

    hls::stream<pixel4_t> stream_blur_simd("blur_simd");
    #pragma HLS STREAM variable=stream_blur_simd depth=64
    #pragma HLS bind_storage variable=stream_blur_simd type=fifo impl=srl

    hls::stream<pixel4_t> stream_sobel_x_simd("sob_x_simd");
    #pragma HLS STREAM variable=stream_sobel_x_simd depth=64
    #pragma HLS bind_storage variable=stream_sobel_x_simd type=fifo impl=srl

    hls::stream<pixel4_t> stream_sobel_y_simd("sob_y_simd");
    #pragma HLS STREAM variable=stream_sobel_y_simd depth=64
    #pragma HLS bind_storage variable=stream_sobel_y_simd type=fifo impl=srl

    hls::stream<pixel4_t> stream_mag_simd("mag_simd");
    #pragma HLS STREAM variable=stream_mag_simd depth=64
    #pragma HLS bind_storage variable=stream_mag_simd type=fifo impl=srl

    hls::stream<pixel4_t> stream_dir_simd("dir_simd");
    #pragma HLS STREAM variable=stream_dir_simd depth=64
    #pragma HLS bind_storage variable=stream_dir_simd type=fifo impl=srl

    hls::stream<pixel4_t> stream_nms_simd("nms_simd");
    #pragma HLS STREAM variable=stream_nms_simd depth=64
    #pragma HLS bind_storage variable=stream_nms_simd type=fifo impl=srl

    hls::stream<pixel4_t> stream_hyst_simd("hyst_simd");
    #pragma HLS STREAM variable=stream_hyst_simd depth=64
    #pragma HLS bind_storage variable=stream_hyst_simd type=fifo impl=srl

    hls::stream<pixel4_t> stream_out_simd("out_simd");
    #pragma HLS STREAM variable=stream_out_simd depth=64
    #pragma HLS bind_storage variable=stream_out_simd type=fifo impl=srl

    read_input_burst(img_r, img_g, img_b, stream_r_simd, stream_g_simd, stream_b_simd);

    kernel1_rgb_to_ycbcr_simd(stream_r_simd, stream_g_simd, stream_b_simd, stream_gray_simd);
    kernel2_median_simd(stream_gray_simd, stream_median_simd);
    kernel3_bilateral_simd(stream_median_simd, stream_blur_simd);
    kernel4_sobel_simd(stream_blur_simd, stream_sobel_x_simd, stream_sobel_y_simd);
    kernel5_mag_dir_simd(stream_sobel_x_simd, stream_sobel_y_simd, stream_mag_simd, stream_dir_simd);
    kernel6_nms_simd(stream_mag_simd, stream_dir_simd, stream_nms_simd);
    kernel7_hysteresis_simd(stream_nms_simd, stream_hyst_simd); 
    kernel8_dilation_simd(stream_hyst_simd, stream_out_simd); 
    
    write_output_burst(stream_out_simd, img_out);
}
