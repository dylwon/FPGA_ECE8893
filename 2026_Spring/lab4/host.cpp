#include <iostream>
#include "dcl.h"

// ---------------------------------------------------------
// Input Initialization
// Generates a deterministic pseudo-random image [0, 255]
// ---------------------------------------------------------
static void init_input(pixel_t in[HEIGHT][WIDTH]) {
    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
            // Pseudo-random generation for testing
            int v = ((r * 113 + c * 59 + 17) & 255); 
            in[r][c] = (pixel_t)v; 
        }
    }
}

// ---------------------------------------------------------
// Golden Reference Kernels (Unoptimized pure C++)
// ---------------------------------------------------------
static void golden_kernel1(pixel_t img_in[HEIGHT][WIDTH], pixel_t stage1[HEIGHT][WIDTH]) {
    coef_t kernel[3][3] = {{0.0625, 0.125, 0.0625}, {0.125, 0.25, 0.125}, {0.0625, 0.125, 0.0625}};
    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
            if (r == 0 || r == HEIGHT - 1 || c == 0 || c == WIDTH - 1) {
                stage1[r][c] = img_in[r][c];
            } else {
                calc_t sum = 0;
                for (int kr = -1; kr <= 1; kr++) {
                    for (int kc = -1; kc <= 1; kc++) {
                        sum += img_in[r + kr][c + kc] * kernel[kr + 1][kc + 1];
                    }
                }
                stage1[r][c] = (pixel_t)sum;
            }
        }
    }
}

static void golden_kernel2(pixel_t stage1[HEIGHT][WIDTH], pixel_t stage2_x[HEIGHT][WIDTH], pixel_t stage2_y[HEIGHT][WIDTH]) {
    coef_t sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    coef_t sobel_y[3][3] = {{ 1, 2, 1}, { 0, 0, 0}, {-1,-2,-1}};
    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
            if (r == 0 || r == HEIGHT - 1 || c == 0 || c == WIDTH - 1) {
                stage2_x[r][c] = 0; stage2_y[r][c] = 0;
            } else {
                calc_t sum_x = 0; calc_t sum_y = 0;
                for (int kr = -1; kr <= 1; kr++) {
                    for (int kc = -1; kc <= 1; kc++) {
                        pixel_t val = stage1[r + kr][c + kc];
                        sum_x += val * sobel_x[kr + 1][kc + 1];
                        sum_y += val * sobel_y[kr + 1][kc + 1];
                    }
                }
                stage2_x[r][c] = (pixel_t)sum_x; stage2_y[r][c] = (pixel_t)sum_y;
            }
        }
    }
}

static void golden_kernel3(pixel_t stage2_x[HEIGHT][WIDTH], pixel_t stage2_y[HEIGHT][WIDTH], pixel_t stage3_mag[HEIGHT][WIDTH], pixel_t stage3_dir[HEIGHT][WIDTH]) {
    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
            calc_t gx = stage2_x[r][c];
            calc_t gy = stage2_y[r][c];
            calc_t abs_gx = (gx < 0) ? (calc_t)(-gx) : gx;
            calc_t abs_gy = (gy < 0) ? (calc_t)(-gy) : gy;
            stage3_mag[r][c] = (pixel_t)(abs_gx + abs_gy);
            stage3_dir[r][c] = (abs_gx > abs_gy) ? (pixel_t)0 : (pixel_t)90;
        }
    }
}

static void golden_kernel4(pixel_t stage3_mag[HEIGHT][WIDTH], pixel_t stage3_dir[HEIGHT][WIDTH], pixel_t stage4[HEIGHT][WIDTH]) {
    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
            if (r == 0 || r == HEIGHT - 1 || c == 0 || c == WIDTH - 1) {
                stage4[r][c] = 0; continue;
            }
            pixel_t mag = stage3_mag[r][c];
            pixel_t dir = stage3_dir[r][c];
            pixel_t mag1 = (dir == 0) ? stage3_mag[r][c - 1] : stage3_mag[r - 1][c];
            pixel_t mag2 = (dir == 0) ? stage3_mag[r][c + 1] : stage3_mag[r + 1][c];
            
            stage4[r][c] = (mag >= mag1 && mag >= mag2) ? mag : (pixel_t)0;
        }
    }
}

static void golden_kernel5(pixel_t stage4[HEIGHT][WIDTH], pixel_t img_out[HEIGHT][WIDTH]) {
    pixel_t HIGH_THRESH = 50;
    pixel_t LOW_THRESH = 20;
    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
            pixel_t val = stage4[r][c];
            if (val >= HIGH_THRESH) img_out[r][c] = 255;
            else if (val >= LOW_THRESH) img_out[r][c] = 127;
            else img_out[r][c] = 0;
        }
    }
}

// ---------------------------------------------------------
// Golden Pipeline Integration
// ---------------------------------------------------------
static void golden_vision_pipeline(pixel_t img_in[HEIGHT][WIDTH], pixel_t img_out[HEIGHT][WIDTH]) {
    static pixel_t stage1[HEIGHT][WIDTH];
    static pixel_t stage2_x[HEIGHT][WIDTH];
    static pixel_t stage2_y[HEIGHT][WIDTH];
    static pixel_t stage3_mag[HEIGHT][WIDTH];
    static pixel_t stage3_dir[HEIGHT][WIDTH];
    static pixel_t stage4[HEIGHT][WIDTH];

    golden_kernel1(img_in, stage1);
    golden_kernel2(stage1, stage2_x, stage2_y);
    golden_kernel3(stage2_x, stage2_y, stage3_mag, stage3_dir);
    golden_kernel4(stage3_mag, stage3_dir, stage4);
    golden_kernel5(stage4, img_out);
}

// ---------------------------------------------------------
// Main Testbench
// ---------------------------------------------------------
int main() {
    // Massive arrays must be static to avoid stack overflow in C-simulation
    static pixel_t in[HEIGHT][WIDTH];
    static pixel_t out_hw[HEIGHT][WIDTH];
    static pixel_t out_gold[HEIGHT][WIDTH];

    init_input(in);

    // Run Device Under Test (Your HLS code)
    std::cout << "Running Hardware Design...\n";
    top_vision_pipeline(in, out_hw);

    // Run Golden Model
    std::cout << "Running Golden Reference...\n";
    golden_vision_pipeline(in, out_gold);

    // Verify Correctness
    int errors = 0;
    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
            // Fixed-point exact match check
            if (out_hw[r][c] != out_gold[r][c]) {
                errors++;
                if (errors <= 10) { // Limit error printing
                    std::cout << "Mismatch at [" << r << "][" << c << "]"
                              << " hw=" << out_hw[r][c].to_double()
                              << " gold=" << out_gold[r][c].to_double()
                              << "\n";
                }
            }
        }
    }

    if (errors == 0) {
        std::cout << "TEST PASSED\n";
        return 0;
    } else {
        std::cout << "TEST FAILED with " << errors << " mismatches\n";
        return 1;
    }
}
