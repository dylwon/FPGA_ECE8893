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
// Kernel 5: Double Thresholding
// ---------------------------------------------------------
void kernel5_double_threshold(pixel_t stage4[HEIGHT][WIDTH], pixel_t img_out[HEIGHT][WIDTH]) {
    pixel_t HIGH_THRESH = 50;
    pixel_t LOW_THRESH = 20;

    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS UNROLL factor=8

            pixel_t val = stage4[r][c];
            if (val >= HIGH_THRESH) {
                img_out[r][c] = 255; 
            } else if (val >= LOW_THRESH) {
                img_out[r][c] = 127; 
            } else {
                img_out[r][c] = 0;   
            }
        }
    }
}

// ---------------------------------------------------------
// Top-Level Function
// ---------------------------------------------------------
void top_vision_pipeline(pixel_t img_in[HEIGHT][WIDTH], pixel_t img_out[HEIGHT][WIDTH]) {
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
    kernel5_double_threshold(stage4, img_out);
}
