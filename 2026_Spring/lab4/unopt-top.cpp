#include "dcl.h"

// ---------------------------------------------------------
// Kernel 0: Full YCbCr Color Space Converter (Baseline)
// ---------------------------------------------------------
void kernel0_rgb_to_ycbcr(
    pixel_t img_r[HEIGHT][WIDTH], 
    pixel_t img_g[HEIGHT][WIDTH], 
    pixel_t img_b[HEIGHT][WIDTH], 
    pixel_t img_gray[HEIGHT][WIDTH]
) {
    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
            
            int red = (int)img_r[r][c];
            int green = (int)img_g[r][c];
            int blue = (int)img_b[r][c];

            // Step 1: ITU-R BT.601 Matrix Multiplication (Bit-shifted by 8)
            int y_calc = (66 * red + 129 * green + 25 * blue + 4096) >> 8;

            // Step 2: Broadcast Legal Saturation
            if (y_calc < 16) y_calc = 16;
            else if (y_calc > 235) y_calc = 235;

            // Step 3: Dynamic Range Normalization (Stretch 16-235 to 0-255)
            int normalized_gray = ((y_calc - 16) * 298) >> 8;

            // Final safety clamp
            if (normalized_gray < 0) normalized_gray = 0;
            else if (normalized_gray > 255) normalized_gray = 255;

            img_gray[r][c] = (pixel_t)normalized_gray;
        }
    }
}

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
// Kernel 3: Magnitude & True Gradient Direction
// Introduces a heavy fixed-point division for every pixel
// ---------------------------------------------------------
void kernel3_magnitude_direction(pixel_t stage2_x[HEIGHT][WIDTH], pixel_t stage2_y[HEIGHT][WIDTH], pixel_t stage3_mag[HEIGHT][WIDTH], pixel_t stage3_dir[HEIGHT][WIDTH]) {
    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
            calc_t gx = stage2_x[r][c];
            calc_t gy = stage2_y[r][c];
            
            // Keep the L1 norm for magnitude
            calc_t abs_gx = (gx < 0) ? (calc_t)(-gx) : gx;
            calc_t abs_gy = (gy < 0) ? (calc_t)(-gy) : gy;
            stage3_mag[r][c] = (pixel_t)(abs_gx + abs_gy);

            // --- THE BOTTLENECK: True Slope Division ---
            pixel_t dir = 0;
            if (gx == 0) {
                // Avoid divide-by-zero
                dir = 90; 
            } else {
                // Massive hardware stall here
                calc_t slope = gy / gx; 

                // Categorize into 0, 45, 90, 135 degrees using tangent approximations
                if (slope > (calc_t)-0.414 && slope <= (calc_t)0.414) {
                    dir = 0;
                } else if (slope > (calc_t)0.414 && slope <= (calc_t)2.414) {
                    dir = 45;
                } else if (slope < (calc_t)-0.414 && slope >= (calc_t)-2.414) {
                    dir = 135;
                } else {
                    dir = 90;
                }
            }
            stage3_dir[r][c] = dir;
        }
    }
}

// ---------------------------------------------------------
// Kernel 4: Non-Maximum Suppression (Corrected Baseline)
// ---------------------------------------------------------
void kernel4_non_max_suppression(pixel_t stage3_mag[HEIGHT][WIDTH], pixel_t stage3_dir[HEIGHT][WIDTH], pixel_t stage4[HEIGHT][WIDTH]) {
    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
            
            // Boundary safety check
            if (r == 0 || r == HEIGHT - 1 || c == 0 || c == WIDTH - 1) {
                stage4[r][c] = 0;
                continue;
            }

            pixel_t mag = stage3_mag[r][c];
            pixel_t dir = stage3_dir[r][c];
            pixel_t mag1 = 0, mag2 = 0;

            if (dir == 0) { 
                // Horizontal edge: check left and right
                mag1 = stage3_mag[r][c - 1];
                mag2 = stage3_mag[r][c + 1];
            } else if (dir == 90) { 
                // Vertical edge: check top and bottom
                mag1 = stage3_mag[r - 1][c];
                mag2 = stage3_mag[r + 1][c];
            } else if (dir == 45) { 
                // Diagonal 45: check bottom-left and top-right
                mag1 = stage3_mag[r + 1][c - 1];
                mag2 = stage3_mag[r - 1][c + 1];
            } else { 
                // Diagonal 135: check top-left and bottom-right
                mag1 = stage3_mag[r - 1][c - 1];
                mag2 = stage3_mag[r + 1][c + 1];
            }

            // Suppress non-maximum pixels
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

            pixel_t local_mean = (pixel_t)(local_sum / (calc_t)25);

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
// Kernel 6: Morphological Dilation (Unoptimized Baseline)
// ---------------------------------------------------------
void kernel6_dilation(pixel_t stage5[HEIGHT][WIDTH], pixel_t img_out[HEIGHT][WIDTH]) {
    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
            
            // Boundary safety margin: 1 pixel for a 3x3 window
            if (r == 0 || r == HEIGHT - 1 || c == 0 || c == WIDTH - 1) {
                img_out[r][c] = 0;
                continue;
            }

            // Dilation logic: 3x3 neighborhood search
            pixel_t max_val = 0;
            for (int kr = -1; kr <= 1; kr++) {
                for (int kc = -1; kc <= 1; kc++) {
                    // If any neighbor is a strong edge (255), the center becomes an edge
                    if (stage5[r + kr][c + kc] == 255) {
                        max_val = 255;
                    }
                }
            }
            
            img_out[r][c] = max_val;
        }
    }
}

// ---------------------------------------------------------
// Top-Level Function
// ---------------------------------------------------------
void top_kernel(
    pixel_t img_r[HEIGHT][WIDTH], 
    pixel_t img_g[HEIGHT][WIDTH], 
    pixel_t img_b[HEIGHT][WIDTH], 
    pixel_t img_out[HEIGHT][WIDTH]
) {
    static pixel_t buf_A[HEIGHT][WIDTH]; 
    static pixel_t buf_B[HEIGHT][WIDTH]; 
    static pixel_t buf_C[HEIGHT][WIDTH]; 
    static pixel_t buf_D[HEIGHT][WIDTH]; 

    kernel0_rgb_to_ycbcr(img_r, img_g, img_b, buf_A); 
    kernel1_gaussian_blur(buf_A, buf_B);         
    kernel2_sobel_gradients(buf_B, buf_A, buf_C);
    kernel3_magnitude_direction(buf_A, buf_C, buf_B, buf_D);
    kernel4_non_max_suppression(buf_B, buf_D, buf_A);
    kernel5_hysteresis(buf_A, buf_B); 
    kernel6_dilation(buf_B, img_out); 
}
