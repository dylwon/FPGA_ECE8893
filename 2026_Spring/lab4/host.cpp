#include <iostream>
#include "dcl.h"

// ---------------------------------------------------------
// Input Initialization (RGB)
// Generates deterministic pseudo-random colored images
// ---------------------------------------------------------
static void init_input_rgb(pixel_t in_r[HEIGHT][WIDTH], pixel_t in_g[HEIGHT][WIDTH], pixel_t in_b[HEIGHT][WIDTH]) {
    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
            // Generate slightly different pseudo-random patterns for each color channel
            in_r[r][c] = (pixel_t)((r * 113 + c * 59 + 17) & 255); 
            in_g[r][c] = (pixel_t)((r * 89 + c * 101 + 31) & 255);
            in_b[r][c] = (pixel_t)((r * 67 + c * 131 + 47) & 255);
        }
    }
}

// ---------------------------------------------------------
// Golden Kernel 0: Full YCbCr Color Space Converter
// ---------------------------------------------------------
static void golden_kernel0(
    pixel_t img_r[HEIGHT][WIDTH], 
    pixel_t img_g[HEIGHT][WIDTH], 
    pixel_t img_b[HEIGHT][WIDTH], 
    pixel_t img_gray[HEIGHT][WIDTH]
) {
    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
            
            // Read incoming RGB pixels
            int red = (int)img_r[r][c];
            int green = (int)img_g[r][c];
            int blue = (int)img_b[r][c];

            // Step 1: ITU-R BT.601 Matrix Multiplication (Bit-shifted by 8)
            int y_calc = (66 * red + 129 * green + 25 * blue + 4096) >> 8;

            // Step 2: Broadcast Legal Saturation
            if (y_calc < 16) {
                y_calc = 16;
            } else if (y_calc > 235) {
                y_calc = 235;
            }

            // Step 3: Dynamic Range Normalization (Stretch 16-235 to 0-255)
            int normalized_gray = ((y_calc - 16) * 298) >> 8;

            // Final safety clamp
            if (normalized_gray < 0) {
                normalized_gray = 0;
            } else if (normalized_gray > 255) {
                normalized_gray = 255;
            }

            img_gray[r][c] = (pixel_t)normalized_gray;
        }
    }
}

// ---------------------------------------------------------
// Golden Kernel 1: 7x7 Gaussian Blur
// ---------------------------------------------------------
static void golden_kernel1(pixel_t img_in[HEIGHT][WIDTH], pixel_t stage1[HEIGHT][WIDTH]) {
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
            if (r < 3 || r >= HEIGHT - 3 || c < 3 || c >= WIDTH - 3) {
                stage1[r][c] = img_in[r][c];
            } else {
                calc_t sum = 0;
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

// ---------------------------------------------------------
// Golden Kernel 3: Magnitude & True Gradient Direction
// ---------------------------------------------------------
static void golden_kernel3(pixel_t stage2_x[HEIGHT][WIDTH], pixel_t stage2_y[HEIGHT][WIDTH], pixel_t stage3_mag[HEIGHT][WIDTH], pixel_t stage3_dir[HEIGHT][WIDTH]) {
    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
            calc_t gx = stage2_x[r][c];
            calc_t gy = stage2_y[r][c];
            
            calc_t abs_gx = (gx < 0) ? (calc_t)(-gx) : gx;
            calc_t abs_gy = (gy < 0) ? (calc_t)(-gy) : gy;
            stage3_mag[r][c] = (pixel_t)(abs_gx + abs_gy);

            pixel_t dir = 0;
            if (gx == 0) {
                dir = 90; 
            } else {
                // The mathematical anchor for the hardware divider
                calc_t slope = gy / gx; 

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

static void golden_kernel4(pixel_t stage3_mag[HEIGHT][WIDTH], pixel_t stage3_dir[HEIGHT][WIDTH], pixel_t stage4[HEIGHT][WIDTH]) {
    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
            // Handle image borders
            if (r == 0 || r == HEIGHT - 1 || c == 0 || c == WIDTH - 1) {
                stage4[r][c] = 0; 
                continue;
            }
            
            pixel_t mag = stage3_mag[r][c];
            pixel_t dir = stage3_dir[r][c];
            pixel_t mag1 = 0, mag2 = 0;
            
            // PROPER NON-MAXIMUM SUPPRESSION (4 Directions)
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
            
            stage4[r][c] = (mag >= mag1 && mag >= mag2) ? mag : (pixel_t)0;
        }
    }
}

// ---------------------------------------------------------
// Golden Kernel 5: Adaptive Thresholding & Hysteresis
// ---------------------------------------------------------
static void golden_kernel5(pixel_t stage4[HEIGHT][WIDTH], pixel_t img_out[HEIGHT][WIDTH]) {
    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
            
            if (r < 2 || r >= HEIGHT - 2 || c < 2 || c >= WIDTH - 2) {
                img_out[r][c] = 0;
                continue;
            }

            calc_t local_sum = 0;
            for (int kr = -2; kr <= 2; kr++) {
                for (int kc = -2; kc <= 2; kc++) {
                    local_sum += stage4[r + kr][c + kc];
                }
            }
            
            pixel_t local_mean = (pixel_t)(local_sum / (calc_t)25); 

            pixel_t HIGH_THRESH = local_mean + (pixel_t)15;
            pixel_t LOW_THRESH  = local_mean - (pixel_t)5;
            
            pixel_t center_pixel = stage4[r][c];
            
            if (center_pixel >= HIGH_THRESH) {
                img_out[r][c] = 255;
            } else if (center_pixel < LOW_THRESH) {
                img_out[r][c] = 0;
            } else {
                bool connected = false;
                for (int kr = -1; kr <= 1; kr++) {
                    for (int kc = -1; kc <= 1; kc++) {
                        if (kr == 0 && kc == 0) continue; 
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
// Kernel 6: Morphological Dilation (Baseline / Golden)
// ---------------------------------------------------------
static void golden_kernel6(pixel_t stage5[HEIGHT][WIDTH], pixel_t img_out[HEIGHT][WIDTH]) {
    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
            
            // Boundary check: skip the outermost pixels
            if (r == 0 || r == HEIGHT - 1 || c == 0 || c == WIDTH - 1) {
                img_out[r][c] = 0;
                continue;
            }

            // Dilation Rule: If ANY pixel in the 3x3 window is an edge (255), 
            // the center pixel becomes an edge.
            pixel_t max_val = 0;
            for (int kr = -1; kr <= 1; kr++) {
                for (int kc = -1; kc <= 1; kc++) {
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
// Golden Pipeline Integration
// ---------------------------------------------------------
static void golden_vision_pipeline(
    pixel_t img_r[HEIGHT][WIDTH], 
    pixel_t img_g[HEIGHT][WIDTH], 
    pixel_t img_b[HEIGHT][WIDTH], 
    pixel_t img_out[HEIGHT][WIDTH]
) {
    // Massive arrays must be static to avoid stack overflow in C-simulation
    static pixel_t stage0_gray[HEIGHT][WIDTH];
    static pixel_t stage1[HEIGHT][WIDTH];
    static pixel_t stage2_x[HEIGHT][WIDTH];
    static pixel_t stage2_y[HEIGHT][WIDTH];
    static pixel_t stage3_mag[HEIGHT][WIDTH];
    static pixel_t stage3_dir[HEIGHT][WIDTH];
    static pixel_t stage4[HEIGHT][WIDTH];
    static pixel_t stage5[HEIGHT][WIDTH];

    golden_kernel0(img_r, img_g, img_b, stage0_gray);
    golden_kernel1(stage0_gray, stage1);
    golden_kernel2(stage1, stage2_x, stage2_y);
    golden_kernel3(stage2_x, stage2_y, stage3_mag, stage3_dir);
    golden_kernel4(stage3_mag, stage3_dir, stage4);
    golden_kernel5(stage4, stage5);
    golden_kernel6(stage5, img_out);
}

// ---------------------------------------------------------
// Main Testbench
// ---------------------------------------------------------
int main() {
    // Massive arrays must be static to avoid stack overflow
    static pixel_t in_r[HEIGHT][WIDTH];
    static pixel_t in_g[HEIGHT][WIDTH];
    static pixel_t in_b[HEIGHT][WIDTH];
    static pixel_t out_hw[HEIGHT][WIDTH];
    static pixel_t out_gold[HEIGHT][WIDTH];

    init_input_rgb(in_r, in_g, in_b);

    std::cout << "Running Hardware Design...\n";
    top_kernel(in_r, in_g, in_b, out_hw);

    std::cout << "Running Golden Reference...\n";
    golden_vision_pipeline(in_r, in_g, in_b, out_gold); // Pass 3 arrays now

    // Verify Correctness with +/- 1% Tolerance
    int errors = 0;
    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
            
            // Convert ap_fixed to double for easy testbench math
            double hw_val = out_hw[r][c].to_double();
            double gold_val = out_gold[r][c].to_double();
            
            // Calculate absolute difference
            double diff = std::abs(hw_val - gold_val);
            
            // Calculate the 1% allowed error threshold
            double allowed_error = 0.01 * std::abs(gold_val);
            
            // Edge Case: If the golden value is exactly 0, 1% of 0 is 0.
            // We need a tiny absolute threshold here to prevent false failures 
            // from negligible hardware noise (like 0.001 != 0.0).
            if (gold_val == 0.0) {
                allowed_error = 0.05; // Small absolute baseline tolerance
            }

            // Check if the difference exceeds the 1% allowed error
            if (diff > allowed_error) {
                errors++;
                if (errors <= 10) { // Limit error printing so it doesn't flood the console
                    std::cout << "Mismatch at [" << r << "][" << c << "]"
                              << " hw=" << hw_val
                              << " gold=" << gold_val
                              << " (diff=" << diff << " > allowed=" << allowed_error << ")\n";
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
