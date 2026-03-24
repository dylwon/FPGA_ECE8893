#include <hls_stream.h>
#include "dcl.h"

// ---------------------------------------------------------
// Kernel 1: Gaussian Blur (Noise Reduction)
// ---------------------------------------------------------
void kernel1_gaussian_blur(pixel_t img_in[HEIGHT][WIDTH], pixel_t stage1[HEIGHT][WIDTH]) {
    coef_t kernel[3][3] = {
        {0.0625, 0.125, 0.0625},
        {0.125,  0.25,  0.125},
        {0.0625, 0.125, 0.0625}
    };
    // Completely partition the 3x3 kernel so all 9 coefficients are read in 1 cycle
    #pragma HLS ARRAY_PARTITION variable=kernel complete dim=0

    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
            // PIPELINE II=1 forces the loop to process a pixel every single clock cycle.
            // UNROLL factor=8 tells the compiler to process 8 pixels simultaneously per cycle.
            #pragma HLS PIPELINE II=1
            #pragma HLS UNROLL factor=8

            if (r == 0 || r == HEIGHT - 1 || c == 0 || c == WIDTH - 1) {
                stage1[r][c] = img_in[r][c];
            } else {
                calc_t sum = 0;
                for (int kr = -1; kr <= 1; kr++) {
                    for (int kc = -1; kc <= 1; kc++) {
                        // The inner loops are automatically unrolled by the PIPELINE directive
                        sum += img_in[r + kr][c + kc] * kernel[kr + 1][kc + 1];
                    }
                }
                stage1[r][c] = (pixel_t)sum;
            }
        }
    }
}

// ---------------------------------------------------------
// Kernel 2: Sobel Gradients (X and Y directions)
// ---------------------------------------------------------
void kernel2_sobel_gradients(pixel_t stage1[HEIGHT][WIDTH], pixel_t stage2_x[HEIGHT][WIDTH], pixel_t stage2_y[HEIGHT][WIDTH]) {
    coef_t sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    coef_t sobel_y[3][3] = {{ 1, 2, 1}, { 0, 0, 0}, {-1,-2,-1}};
    #pragma HLS ARRAY_PARTITION variable=sobel_x complete dim=0
    #pragma HLS ARRAY_PARTITION variable=sobel_y complete dim=0

    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS UNROLL factor=8

            if (r == 0 || r == HEIGHT - 1 || c == 0 || c == WIDTH - 1) {
                stage2_x[r][c] = 0;
                stage2_y[r][c] = 0;
            } else {
                calc_t sum_x = 0;
                calc_t sum_y = 0;
                for (int kr = -1; kr <= 1; kr++) {
                    for (int kc = -1; kc <= 1; kc++) {
                        pixel_t val = stage1[r + kr][c + kc];
                        sum_x += val * sobel_x[kr + 1][kc + 1];
                        sum_y += val * sobel_y[kr + 1][kc + 1];
                    }
                }
                stage2_x[r][c] = (pixel_t)sum_x;
                stage2_y[r][c] = (pixel_t)sum_y;
            }
        }
    }
}

// ---------------------------------------------------------
// Kernel 3: Gradient Magnitude & Approximate Direction
// ---------------------------------------------------------
void kernel3_magnitude_direction(pixel_t stage2_x[HEIGHT][WIDTH], pixel_t stage2_y[HEIGHT][WIDTH], pixel_t stage3_mag[HEIGHT][WIDTH], pixel_t stage3_dir[HEIGHT][WIDTH]) {
    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS UNROLL factor=8

            calc_t gx = stage2_x[r][c];
            calc_t gy = stage2_y[r][c];
            
            calc_t abs_gx = (gx < 0) ? (calc_t)(-gx) : gx;
            calc_t abs_gy = (gy < 0) ? (calc_t)(-gy) : gy;
            stage3_mag[r][c] = (pixel_t)(abs_gx + abs_gy);

            stage3_dir[r][c] = (abs_gx > abs_gy) ? (pixel_t)0 : (pixel_t)90;
        }
    }
}

// ---------------------------------------------------------
// Kernel 4: Non-Maximum Suppression (Thinning)
// ---------------------------------------------------------
void kernel4_non_max_suppression(pixel_t stage3_mag[HEIGHT][WIDTH], pixel_t stage3_dir[HEIGHT][WIDTH], pixel_t stage4[HEIGHT][WIDTH]) {
    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS UNROLL factor=8

            if (r == 0 || r == HEIGHT - 1 || c == 0 || c == WIDTH - 1) {
                stage4[r][c] = 0;
                continue;
            }

            pixel_t mag = stage3_mag[r][c];
            pixel_t dir = stage3_dir[r][c];
            pixel_t mag1 = 0, mag2 = 0;

            if (dir == 0) { 
                mag1 = stage3_mag[r][c - 1];
                mag2 = stage3_mag[r][c + 1];
            } else { 
                mag1 = stage3_mag[r - 1][c];
                mag2 = stage3_mag[r + 1][c];
            }

            stage4[r][c] = (mag >= mag1 && mag >= mag2) ? mag : (pixel_t)0;
        }
    }
}

// ---------------------------------------------------------
// OPTIMIZED Kernel 5: Hysteresis with Line Buffers
// ---------------------------------------------------------
void kernel5_hysteresis_opt(hls::stream<pixel_t>& stream_in, hls::stream<pixel_t>& stream_out) {
    pixel_t HIGH_THRESH = 50;
    pixel_t LOW_THRESH = 20;

    // 1. The Memory Architecture: 2 Rows of Line Buffers
    static pixel_t line_buf[2][WIDTH];
    // Partition dim=1 splits this into 2 completely independent BRAMs/Registers
    // so we can read from both simultaneously in 1 clock cycle.
    #pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1

    // 2. The 3x3 Sliding Window (Shift Registers)
    pixel_t window[3][3];
    // Completely partition the window into individual flip-flops
    #pragma HLS ARRAY_PARTITION variable=window complete dim=0

    // We must loop slightly further to "flush" the final pixels out of the window
    for (int r = 0; r < HEIGHT + 1; r++) {
        for (int c = 0; c < WIDTH + 1; c++) {
            
            // THE HOLY GRAIL PRAGMA: Forces 1 pixel processed per clock cycle
            #pragma HLS PIPELINE II=1

            // --- A. SHIFT THE WINDOW LEFT ---
            for (int i = 0; i < 3; i++) {
                window[i][0] = window[i][1];
                window[i][1] = window[i][2];
            }

            // --- B. READ NEW PIXEL & UPDATE BUFFERS ---
            pixel_t new_pix = 0;
            // Only read from the stream if we are within the actual image bounds
            if (r < HEIGHT && c < WIDTH) {
                new_pix = stream_in.read();
            }

            // Fetch the old pixels from the same column in the previous two rows
            pixel_t top_pixel = line_buf[0][c];
            pixel_t mid_pixel = line_buf[1][c];

            // Push the new column into the right side of the 3x3 window
            window[0][2] = top_pixel;
            window[1][2] = mid_pixel;
            window[2][2] = new_pix;

            // Update the line buffers for the next row
            if (c < WIDTH) {
                line_buf[0][c] = mid_pixel;
                line_buf[1][c] = new_pix;
            }

            // --- C. COMPUTE HYSTERESIS (Delayed by 1 Row & 1 Col) ---
            // Because the center of our 3x3 window is at window[1][1], 
            // the pixel we are actually evaluating is at (r-1, c-1).
            if (r >= 1 && r <= HEIGHT && c >= 1 && c <= WIDTH) {
                
                int out_r = r - 1;
                int out_c = c - 1;
                pixel_t center_pixel = window[1][1];
                pixel_t result = 0;

                // Handle boundaries
                if (out_r == 0 || out_r == HEIGHT - 1 || out_c == 0 || out_c == WIDTH - 1) {
                    result = 0;
                } 
                else if (center_pixel >= HIGH_THRESH) {
                    result = 255; // Definitively strong
                } 
                else if (center_pixel < LOW_THRESH) {
                    result = 0;   // Definitively weak
                } 
                else {
                    // It is weak. Check the 8 surrounding registers in the window
                    bool connected = false;
                    for (int kr = 0; kr < 3; kr++) {
                        for (int kc = 0; kc < 3; kc++) {
                            if (kr == 1 && kc == 1) continue; // Skip center
                            if (window[kr][kc] >= HIGH_THRESH) {
                                connected = true;
                            }
                        }
                    }
                    result = connected ? (pixel_t)255 : (pixel_t)0;
                }

                // Push the final result to the output stream
                stream_out.write(result);
            }
        }
    }
}

// ---------------------------------------------------------
// Top-Level Function
// ---------------------------------------------------------
void top_kernel(pixel_t img_in[HEIGHT][WIDTH], pixel_t img_out[HEIGHT][WIDTH]) {
    // 1. DATAFLOW Directive: Overlaps the execution of all 5 kernels.
    #pragma HLS DATAFLOW

    // 2. Intermediate Memory Declarations (Ping-Pong BRAMs)
    static pixel_t stage1[HEIGHT][WIDTH];
    static pixel_t stage2_x[HEIGHT][WIDTH];
    static pixel_t stage2_y[HEIGHT][WIDTH];
    static pixel_t stage3_mag[HEIGHT][WIDTH];
    static pixel_t stage3_dir[HEIGHT][WIDTH];
    static pixel_t stage4[HEIGHT][WIDTH];

    // 3. ARRAY PARTITIONING: The Secret to High Frequency and Unrolling.
    // Standard Block RAMs only have 2 read/write ports. To process 8 pixels at once, 
    // we must split these arrays into 8 separate physical memory banks.
    #pragma HLS ARRAY_PARTITION variable=img_in cyclic factor=8 dim=2
    #pragma HLS ARRAY_PARTITION variable=stage1 cyclic factor=8 dim=2
    #pragma HLS ARRAY_PARTITION variable=stage2_x cyclic factor=8 dim=2
    #pragma HLS ARRAY_PARTITION variable=stage2_y cyclic factor=8 dim=2
    #pragma HLS ARRAY_PARTITION variable=stage3_mag cyclic factor=8 dim=2
    #pragma HLS ARRAY_PARTITION variable=stage3_dir cyclic factor=8 dim=2
    #pragma HLS ARRAY_PARTITION variable=stage4 cyclic factor=8 dim=2
    #pragma HLS ARRAY_PARTITION variable=img_out cyclic factor=8 dim=2

    kernel1_gaussian_blur(img_in, stage1);
    kernel2_sobel_gradients(stage1, stage2_x, stage2_y);
    kernel3_magnitude_direction(stage2_x, stage2_y, stage3_mag, stage3_dir);
    kernel4_non_max_suppression(stage3_mag, stage3_dir, stage4);
    kernel5_hysteresis_opt(stage4, img_out);
}
