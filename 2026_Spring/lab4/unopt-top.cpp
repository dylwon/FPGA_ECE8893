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
// Kernel 5: Double Thresholding & Edge Tracking (Hysteresis)
// Categorizes pixels and connects weak edges to strong edges.
// ---------------------------------------------------------
void kernel5_hysteresis(pixel_t stage4[HEIGHT][WIDTH], pixel_t img_out[HEIGHT][WIDTH]) {
    pixel_t HIGH_THRESH = 50;
    pixel_t LOW_THRESH = 20;

    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
            
            // Boundary pixels are set to zero to avoid out-of-bounds memory access
            if (r == 0 || r == HEIGHT - 1 || c == 0 || c == WIDTH - 1) {
                img_out[r][c] = 0;
                continue;
            }

            pixel_t center_pixel = stage4[r][c];
            
            // Step 1: Is it a definitively strong edge?
            if (center_pixel >= HIGH_THRESH) {
                img_out[r][c] = 255;
            } 
            // Step 2: Is it definitively NOT an edge?
            else if (center_pixel < LOW_THRESH) {
                img_out[r][c] = 0;
            } 
            // Step 3: It is a weak edge. Perform 8-way neighborhood Hysteresis.
            else {
                bool connected_to_strong = false;
                
                // Check the 3x3 neighborhood around the weak pixel
                for (int kr = -1; kr <= 1; kr++) {
                    for (int kc = -1; kc <= 1; kc++) {
                        
                        // Skip checking the center pixel against itself
                        if (kr == 0 && kc == 0) continue; 
                        
                        pixel_t neighbor = stage4[r + kr][c + kc];
                        
                        // If any neighbor is a strong edge, flag it
                        if (neighbor >= HIGH_THRESH) {
                            connected_to_strong = true;
                        }
                    }
                }
                
                // Promote to strong edge if connected, otherwise suppress to zero
                if (connected_to_strong) {
                    img_out[r][c] = 255;
                } else {
                    img_out[r][c] = 0;
                }
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
