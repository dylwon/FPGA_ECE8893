#include "dcl.h"

// ---------------------------------------------------------
// Kernel 1: 7x7 Gaussian Blur (High-Fidelity Noise Reduction)
// Massively increases baseline cycle count via 49 MACs per pixel.
// ---------------------------------------------------------
void kernel1_gaussian_blur(pixel_t img_in[HEIGHT][WIDTH], pixel_t stage1[HEIGHT][WIDTH]) {
    // 7x7 Gaussian Kernel Approximation (Sum = ~1.0)
    coef_t kernel[7][7] = {
        {0.0000, 0.0002, 0.0011, 0.0018, 0.0011, 0.0002, 0.0000},
        {0.0002, 0.0029, 0.0130, 0.0215, 0.0130, 0.0029, 0.0002},
        {0.0011, 0.0130, 0.0585, 0.0965, 0.0585, 0.0130, 0.0011},
        {0.0018, 0.0215, 0.0965, 0.1591, 0.0965, 0.0215, 0.0018},
        {0.0011, 0.0130, 0.0585, 0.0965, 0.0585, 0.0130, 0.0011},
        {0.0002, 0.0029, 0.0130, 0.0215, 0.0130, 0.0029, 0.0002},
        {0.0000, 0.0002, 0.0011, 0.0018, 0.0011, 0.0002, 0.0000}
    };

    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
            
            // Boundary safety margin is now 3 pixels for a 7x7 window
            if (r < 3 || r >= HEIGHT - 3 || c < 3 || c >= WIDTH - 3) {
                stage1[r][c] = img_in[r][c];
            } else {
                calc_t sum = 0;
                
                // Deeply nested 7x7 loop (The CPU Bottleneck)
                for (int kr = -3; kr <= 3; kr++) {
                    for (int kc = -3; kc <= 3; kc++) {
                        sum += img_in[r + kr][c + kc] * kernel[kr + 3][kc + 3];
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

    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
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
            calc_t gx = stage2_x[r][c];
            calc_t gy = stage2_y[r][c];
            
            calc_t abs_gx = (gx < 0) ? (calc_t)(-gx) : gx;
            calc_t abs_gy = (gy < 0) ? (calc_t)(-gy) : gy;
            stage3_mag[r][c] = (pixel_t)(abs_gx + abs_gy);

            pixel_t dir = 0;
            if (abs_gx > abs_gy) {
                dir = 0; 
            } else {
                dir = 90; 
            }
            stage3_dir[r][c] = dir;
        }
    }
}

// ---------------------------------------------------------
// Kernel 4: Non-Maximum Suppression (Thinning)
// ---------------------------------------------------------
void kernel4_non_max_suppression(pixel_t stage3_mag[HEIGHT][WIDTH], pixel_t stage3_dir[HEIGHT][WIDTH], pixel_t stage4[HEIGHT][WIDTH]) {
    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
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

            if (mag >= mag1 && mag >= mag2) {
                stage4[r][c] = mag;
            } else {
                stage4[r][c] = 0;
            }
        }
    }
}

// ---------------------------------------------------------
// Kernel 5: Adaptive Thresholding & Edge Tracking (Hysteresis)
// Introduces a 5x5 local mean calculation before hysteresis.
// ---------------------------------------------------------
void kernel5_hysteresis(pixel_t stage4[HEIGHT][WIDTH], pixel_t img_out[HEIGHT][WIDTH]) {
    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
            
            // Boundary safety margin is 2 pixels for a 5x5 window
            if (r < 2 || r >= HEIGHT - 2 || c < 2 || c >= WIDTH - 2) {
                img_out[r][c] = 0;
                continue;
            }

            // --- Step 1: Calculate 5x5 Local Mean ---
            calc_t local_sum = 0;
            for (int kr = -2; kr <= 2; kr++) {
                for (int kc = -2; kc <= 2; kc++) {
                    local_sum += stage4[r + kr][c + kc];
                }
            }
            // Divide by 25 (Multiply by 0.04 to avoid a heavy hardware divider)
            pixel_t local_mean = (pixel_t)(local_sum * (calc_t)0.04); 

            // --- Step 2: Set Dynamic Thresholds ---
            pixel_t HIGH_THRESH = local_mean + (pixel_t)15;
            pixel_t LOW_THRESH  = local_mean - (pixel_t)5;
            
            pixel_t center_pixel = stage4[r][c];
            
            // --- Step 3: Hysteresis Logic ---
            if (center_pixel >= HIGH_THRESH) {
                img_out[r][c] = 255;
            } else if (center_pixel < LOW_THRESH) {
                img_out[r][c] = 0;
            } else {
                bool connected = false;
                // Standard 3x3 check for connected strong edges
                for (int kr = -1; kr <= 1; kr++) {
                    for (int kc = -1; kc <= 1; kc++) {
                        if (kr == 0 && kc == 0) continue; 
                        
                        // It must be connected to a pixel that is ALSO above the new dynamic high threshold
                        if (stage4[r + kr][c + kc] >= HIGH_THRESH) {
                            connected = true;
                        }
                    }
                }
                img_out[r][c] = connected ? (pixel_t)255 : (pixel_t)0;
            }
        }
    }
}

// ---------------------------------------------------------
// Top-Level Function
// ---------------------------------------------------------
void top_kernel(pixel_t img_in[HEIGHT][WIDTH], pixel_t img_out[HEIGHT][WIDTH]) {
    static pixel_t stage1[HEIGHT][WIDTH];
    static pixel_t stage2_x[HEIGHT][WIDTH];
    static pixel_t stage2_y[HEIGHT][WIDTH];
    static pixel_t stage3_mag[HEIGHT][WIDTH];
    static pixel_t stage3_dir[HEIGHT][WIDTH];
    static pixel_t stage4[HEIGHT][WIDTH];

    kernel1_gaussian_blur(img_in, stage1);
    kernel2_sobel_gradients(stage1, stage2_x, stage2_y);
    kernel3_magnitude_direction(stage2_x, stage2_y, stage3_mag, stage3_dir);
    kernel4_non_max_suppression(stage3_mag, stage3_dir, stage4);
    kernel5_hysteresis(stage4, img_out);
}
