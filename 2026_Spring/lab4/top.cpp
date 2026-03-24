#include "dcl.h"
#include <hls_stream.h>

// ---------------------------------------------------------
// Helper: Array to Stream / Stream to Array
// ---------------------------------------------------------
void read_input(pixel_t img_in[HEIGHT][WIDTH], hls::stream<pixel_t>& stream_out) {
    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
            #pragma HLS PIPELINE II=1
            stream_out.write(img_in[r][c]);
        }
    }
}

void write_output(hls::stream<pixel_t>& stream_in, pixel_t img_out[HEIGHT][WIDTH]) {
    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
            #pragma HLS PIPELINE II=1
            img_out[r][c] = stream_in.read();
        }
    }
}

// ---------------------------------------------------------
// Kernel 1: 7x7 Gaussian Blur
// ---------------------------------------------------------
void kernel1_gaussian_blur_opt(hls::stream<pixel_t>& stream_in, hls::stream<pixel_t>& stream_out) {
    coef_t kernel[7][7] = {
        {0.0000, 0.0002, 0.0011, 0.0018, 0.0011, 0.0002, 0.0000},
        {0.0002, 0.0029, 0.0130, 0.0215, 0.0130, 0.0029, 0.0002},
        {0.0011, 0.0130, 0.0585, 0.0965, 0.0585, 0.0130, 0.0011},
        {0.0018, 0.0215, 0.0965, 0.1591, 0.0965, 0.0215, 0.0018},
        {0.0011, 0.0130, 0.0585, 0.0965, 0.0585, 0.0130, 0.0011},
        {0.0002, 0.0029, 0.0130, 0.0215, 0.0130, 0.0029, 0.0002},
        {0.0000, 0.0002, 0.0011, 0.0018, 0.0011, 0.0002, 0.0000}
    };
    #pragma HLS ARRAY_PARTITION variable=kernel complete dim=0

    static pixel_t line_buf[6][WIDTH];
    #pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1
    pixel_t window[7][7];
    #pragma HLS ARRAY_PARTITION variable=window complete dim=0

    for (int r = 0; r < HEIGHT + 3; r++) {
        for (int c = 0; c < WIDTH + 3; c++) {
            #pragma HLS PIPELINE II=1

            pixel_t new_pix = 0;
            if (r < HEIGHT && c < WIDTH) new_pix = stream_in.read();

            // Shift Window
            for (int i = 0; i < 7; i++) {
                for (int j = 0; j < 6; j++) window[i][j] = window[i][j + 1];
            }

            // Load from Line Buffers using a safe temp column (fixes memory dependencies)
            pixel_t col[6];
            #pragma HLS ARRAY_PARTITION variable=col complete dim=0
            int c_idx = (c < WIDTH) ? c : WIDTH - 1;
            for (int i = 0; i < 6; i++) {
                col[i] = line_buf[i][c_idx];
                window[i][6] = col[i];
            }
            window[6][6] = new_pix;

            // Update Line Buffers
            if (c < WIDTH) {
                for (int i = 0; i < 5; i++) line_buf[i][c] = col[i + 1];
                line_buf[5][c] = new_pix;
            }

            // Output Phase
            if (r >= 3 && c >= 3) {
                int out_r = r - 3; int out_c = c - 3;
                if (out_r < 3 || out_r >= HEIGHT - 3 || out_c < 3 || out_c >= WIDTH - 3) {
                    stream_out.write(window[3][3]); 
                } else {
                    calc_t sum = 0;
                    for (int kr = 0; kr < 7; kr++) {
                        for (int kc = 0; kc < 7; kc++) sum += window[kr][kc] * kernel[kr][kc];
                    }
                    stream_out.write((pixel_t)sum);
                }
            }
        }
    }
}

// ---------------------------------------------------------
// Kernel 2: Sobel Gradients
// ---------------------------------------------------------
void kernel2_sobel_opt(hls::stream<pixel_t>& stream_in, hls::stream<pixel_t>& stream_x, hls::stream<pixel_t>& stream_y) {
    coef_t sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    coef_t sobel_y[3][3] = {{ 1, 2, 1}, { 0, 0, 0}, {-1,-2,-1}};
    #pragma HLS ARRAY_PARTITION variable=sobel_x complete dim=0
    #pragma HLS ARRAY_PARTITION variable=sobel_y complete dim=0

    static pixel_t line_buf[2][WIDTH];
    #pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1
    pixel_t window[3][3];
    #pragma HLS ARRAY_PARTITION variable=window complete dim=0

    for (int r = 0; r < HEIGHT + 1; r++) {
        for (int c = 0; c < WIDTH + 1; c++) {
            #pragma HLS PIPELINE II=1

            pixel_t new_pix = 0;
            if (r < HEIGHT && c < WIDTH) new_pix = stream_in.read();

            for (int i = 0; i < 3; i++) {
                window[i][0] = window[i][1]; window[i][1] = window[i][2];
            }

            pixel_t col[2];
            #pragma HLS ARRAY_PARTITION variable=col complete dim=0
            int c_idx = (c < WIDTH) ? c : WIDTH - 1;
            col[0] = line_buf[0][c_idx]; col[1] = line_buf[1][c_idx];
            
            window[0][2] = col[0]; window[1][2] = col[1]; window[2][2] = new_pix;

            if (c < WIDTH) {
                line_buf[0][c] = col[1];
                line_buf[1][c] = new_pix;
            }

            if (r >= 1 && c >= 1) {
                int out_r = r - 1; int out_c = c - 1;
                if (out_r == 0 || out_r == HEIGHT - 1 || out_c == 0 || out_c == WIDTH - 1) {
                    stream_x.write(0); stream_y.write(0);
                } else {
                    calc_t sum_x = 0; calc_t sum_y = 0;
                    for (int kr = 0; kr < 3; kr++) {
                        for (int kc = 0; kc < 3; kc++) {
                            sum_x += window[kr][kc] * sobel_x[kr][kc];
                            sum_y += window[kr][kc] * sobel_y[kr][kc];
                        }
                    }
                    stream_x.write((pixel_t)sum_x); stream_y.write((pixel_t)sum_y);
                }
            }
        }
    }
}

// ---------------------------------------------------------
// Kernel 3: Magnitude & Direction
// ---------------------------------------------------------
void kernel3_mag_dir_opt(hls::stream<pixel_t>& stream_x, hls::stream<pixel_t>& stream_y, hls::stream<pixel_t>& stream_mag, hls::stream<pixel_t>& stream_dir) {
    // 1-to-1 processing. No window/delay required.
    for (int i = 0; i < HEIGHT * WIDTH; i++) {
        #pragma HLS PIPELINE II=1

        calc_t gx = stream_x.read();
        calc_t gy = stream_y.read();
        
        calc_t abs_gx = (gx < 0) ? (calc_t)(-gx) : gx;
        calc_t abs_gy = (gy < 0) ? (calc_t)(-gy) : gy;
        stream_mag.write((pixel_t)(abs_gx + abs_gy));

        pixel_t dir = 0;
        if (gx == 0) {
            dir = 90; 
        } else {
            calc_t slope = gy / gx; 
            if (slope > (calc_t)-0.414 && slope <= (calc_t)0.414) dir = 0;
            else if (slope > (calc_t)0.414 && slope <= (calc_t)2.414) dir = 45;
            else if (slope < (calc_t)-0.414 && slope >= (calc_t)-2.414) dir = 135;
            else dir = 90;
        }
        stream_dir.write(dir);
    }
}

// ---------------------------------------------------------
// Kernel 4: Non-Maximum Suppression
// ---------------------------------------------------------
void kernel4_nms_opt(hls::stream<pixel_t>& stream_mag, hls::stream<pixel_t>& stream_dir, hls::stream<pixel_t>& stream_out) {
    static pixel_t mag_buf[2][WIDTH];
    static pixel_t dir_buf[2][WIDTH];
    #pragma HLS ARRAY_PARTITION variable=mag_buf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=dir_buf complete dim=1

    pixel_t mag_win[3][3], dir_win[3][3];
    #pragma HLS ARRAY_PARTITION variable=mag_win complete dim=0
    #pragma HLS ARRAY_PARTITION variable=dir_win complete dim=0

    for (int r = 0; r < HEIGHT + 1; r++) {
        for (int c = 0; c < WIDTH + 1; c++) {
            #pragma HLS PIPELINE II=1

            pixel_t new_mag = 0, new_dir = 0;
            if (r < HEIGHT && c < WIDTH) {
                new_mag = stream_mag.read(); new_dir = stream_dir.read();
            }

            for (int i = 0; i < 3; i++) {
                mag_win[i][0] = mag_win[i][1]; mag_win[i][1] = mag_win[i][2];
                dir_win[i][0] = dir_win[i][1]; dir_win[i][1] = dir_win[i][2];
            }

            pixel_t col_mag[2], col_dir[2];
            #pragma HLS ARRAY_PARTITION variable=col_mag complete dim=0
            #pragma HLS ARRAY_PARTITION variable=col_dir complete dim=0
            int c_idx = (c < WIDTH) ? c : WIDTH - 1;
            col_mag[0] = mag_buf[0][c_idx]; col_mag[1] = mag_buf[1][c_idx];
            col_dir[0] = dir_buf[0][c_idx]; col_dir[1] = dir_buf[1][c_idx];

            mag_win[0][2] = col_mag[0]; mag_win[1][2] = col_mag[1]; mag_win[2][2] = new_mag;
            dir_win[0][2] = col_dir[0]; dir_win[1][2] = col_dir[1]; dir_win[2][2] = new_dir;

            if (c < WIDTH) {
                mag_buf[0][c] = col_mag[1]; mag_buf[1][c] = new_mag;
                dir_buf[0][c] = col_dir[1]; dir_buf[1][c] = new_dir;
            }

            if (r >= 1 && c >= 1) {
                int out_r = r - 1; int out_c = c - 1;
                if (out_r == 0 || out_r == HEIGHT - 1 || out_c == 0 || out_c == WIDTH - 1) {
                    stream_out.write(0);
                } else {
                    pixel_t mag = mag_win[1][1];
                    pixel_t dir = dir_win[1][1];
                    pixel_t mag1 = 0, mag2 = 0;
                    
                    if (dir == 0) { mag1 = mag_win[1][0]; mag2 = mag_win[1][2]; }
                    else if (dir == 90) { mag1 = mag_win[0][1]; mag2 = mag_win[2][1]; }
                    else if (dir == 45) { mag1 = mag_win[2][0]; mag2 = mag_win[0][2]; }
                    else { mag1 = mag_win[0][0]; mag2 = mag_win[2][2]; }

                    stream_out.write((mag >= mag1 && mag >= mag2) ? mag : (pixel_t)0);
                }
            }
        }
    }
}

// ---------------------------------------------------------
// Kernel 5: Adaptive Hysteresis
// ---------------------------------------------------------
void kernel5_hysteresis_opt(hls::stream<pixel_t>& stream_in, hls::stream<pixel_t>& stream_out) {
    static pixel_t line_buf[4][WIDTH];
    #pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1
    pixel_t window[5][5];
    #pragma HLS ARRAY_PARTITION variable=window complete dim=0

    for (int r = 0; r < HEIGHT + 2; r++) {
        for (int c = 0; c < WIDTH + 2; c++) {
            #pragma HLS PIPELINE II=1

            pixel_t new_pix = 0;
            if (r < HEIGHT && c < WIDTH) new_pix = stream_in.read();

            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 4; j++) window[i][j] = window[i][j + 1];
            }

            pixel_t col[4];
            #pragma HLS ARRAY_PARTITION variable=col complete dim=0
            int c_idx = (c < WIDTH) ? c : WIDTH - 1;
            for (int i = 0; i < 4; i++) {
                col[i] = line_buf[i][c_idx];
                window[i][4] = col[i];
            }
            window[4][4] = new_pix;

            if (c < WIDTH) {
                for (int i = 0; i < 3; i++) line_buf[i][c] = col[i + 1];
                line_buf[3][c] = new_pix;
            }

            if (r >= 2 && c >= 2) {
                int out_r = r - 2; int out_c = c - 2;
                
                if (out_r < 2 || out_r >= HEIGHT - 2 || out_c < 2 || out_c >= WIDTH - 2) {
                    stream_out.write(0);
                } else {
                    calc_t local_sum = 0;
                    for (int kr = 0; kr < 5; kr++) {
                        for (int kc = 0; kc < 5; kc++) local_sum += window[kr][kc];
                    }
                    
                    pixel_t local_mean = (pixel_t)(local_sum / (calc_t)25); 

                    pixel_t HIGH_THRESH = local_mean + (pixel_t)15;
                    pixel_t LOW_THRESH  = local_mean - (pixel_t)5;
                    pixel_t center_pixel = window[2][2];
                    
                    if (center_pixel >= HIGH_THRESH) stream_out.write(255);
                    else if (center_pixel < LOW_THRESH) stream_out.write(0);
                    else {
                        bool connected = false;
                        for (int kr = 1; kr <= 3; kr++) {
                            for (int kc = 1; kc <= 3; kc++) {
                                if (kr == 2 && kc == 2) continue; 
                                if (window[kr][kc] >= HIGH_THRESH) connected = true;
                            }
                        }
                        stream_out.write(connected ? (pixel_t)255 : (pixel_t)0);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------
// Top-Level Function
// ---------------------------------------------------------
void top_kernel(pixel_t img_in[HEIGHT][WIDTH], pixel_t img_out[HEIGHT][WIDTH]) {
    #pragma HLS DATAFLOW

    // Instantiate elastic FIFOs
    hls::stream<pixel_t> stream_in("in");
    hls::stream<pixel_t> stream_k1("k1");
    hls::stream<pixel_t> stream_k2x("k2x");
    hls::stream<pixel_t> stream_k2y("k2y");
    hls::stream<pixel_t> stream_k3mag("k3mag");
    hls::stream<pixel_t> stream_k3dir("k3dir");
    hls::stream<pixel_t> stream_k4("k4");
    hls::stream<pixel_t> stream_out("out");

    // --- THE FIX: Elastic Buffering ---
    // Expand depth to 512 to swallow row-blanking phase drifts.
    // This consumes virtually zero FPGA resources (just a few LUTRAMs) 
    // but completely eliminates DATAFLOW deadlocks.
    #pragma HLS STREAM variable=stream_in depth=512
    #pragma HLS STREAM variable=stream_k1 depth=512
    #pragma HLS STREAM variable=stream_k2x depth=512
    #pragma HLS STREAM variable=stream_k2y depth=512
    #pragma HLS STREAM variable=stream_k3mag depth=512
    #pragma HLS STREAM variable=stream_k3dir depth=512
    #pragma HLS STREAM variable=stream_k4 depth=512
    #pragma HLS STREAM variable=stream_out depth=512

    // Wire the hardware blocks together
    read_input(img_in, stream_in);
    kernel1_gaussian_blur_opt(stream_in, stream_k1);
    kernel2_sobel_opt(stream_k1, stream_k2x, stream_k2y);
    kernel3_mag_dir_opt(stream_k2x, stream_k2y, stream_k3mag, stream_k3dir);
    kernel4_nms_opt(stream_k3mag, stream_k3dir, stream_k4);
    kernel5_hysteresis_opt(stream_k4, stream_out);
    write_output(stream_out, img_out);
}
