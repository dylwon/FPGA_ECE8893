#include <iostream>
#include "dcl.h"

// ---------------------------------------------------------
// Input Initialization for RGB
// Generates randomized colored images
// ---------------------------------------------------------
static void init_input_rgb(pixel_t in_r[HEIGHT][WIDTH], pixel_t in_g[HEIGHT][WIDTH], pixel_t in_b[HEIGHT][WIDTH]) {
    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
            // Generate random patterns for each color channel
            in_r[r][c] = (pixel_t)((r * 113 + c * 59 + 17) & 255); 
            in_g[r][c] = (pixel_t)((r * 89 + c * 101 + 31) & 255);
            in_b[r][c] = (pixel_t)((r * 67 + c * 131 + 47) & 255);
        }
    }
}

// -------------------------------------------------------------------------
// Golden Kernel 1: Full YCbCr Color Space Converter (ITU-R BT.601 standard)
// -------------------------------------------------------------------------
static void golden_kernel1(
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

            int y = (66 * red + 129 * green + 25 * blue + 4096) >> 8;

            if (y < 16) {
                y = 16;
            } else if (y > 235) {
                y = 235;
            }

            int normalized_gray = ((y - 16) * 298) >> 8;

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
// Golden Kernel 2: Median Filter 
// ---------------------------------------------------------
static void golden_kernel2(
    pixel_t img_in[HEIGHT][WIDTH], 
    pixel_t img_out[HEIGHT][WIDTH]
) {
    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
            
            if (r < 2 || r >= HEIGHT - 2 || c < 2 || c >= WIDTH - 2) {
                img_out[r][c] = img_in[r][c];
                continue;
            }

            pixel_t flat_window[25];
            int idx = 0;
            for (int kr = -2; kr <= 2; kr++) {
                for (int kc = -2; kc <= 2; kc++) {
                    flat_window[idx++] = img_in[r + kr][c + kc];
                }
            }

            for (int i = 0; i < 24; i++) {
                for (int j = 0; j < 24 - i; j++) {
                    if (flat_window[j] > flat_window[j + 1]) {
                        pixel_t temp = flat_window[j];
                        flat_window[j] = flat_window[j + 1];
                        flat_window[j + 1] = temp;
                    }
                }
            }

            img_out[r][c] = flat_window[12];
        }
    }
}


// ---------------------------------------------------------
// Golden Kernel 3: Bilateral Filter 
// ---------------------------------------------------------
static void golden_kernel3(pixel_t img_in[HEIGHT][WIDTH], pixel_t img_out[HEIGHT][WIDTH]) {
    
    const int spatial_w[5][5] = {
        { 1,  4,  7,  4,  1},
        { 4, 16, 26, 16,  4},
        { 7, 26, 41, 26,  7},
        { 4, 16, 26, 16,  4},
        { 1,  4,  7,  4,  1}
    };

    const int color_w_lut[32] = {
        255, 245, 220, 183, 141, 101, 67, 41, 23, 12, 6, 3, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
            
            if (r < 2 || r >= HEIGHT - 2 || c < 2 || c >= WIDTH - 2) {
                img_out[r][c] = img_in[r][c];
                continue;
            }

            int center_val = (int)img_in[r][c];
            int val_sum = 0;
            int weight_sum = 0;

            for (int kr = -2; kr <= 2; kr++) {
                for (int kc = -2; kc <= 2; kc++) {
                    int neighbor_val = (int)img_in[r + kr][c + kc];
                    
                    int diff = std::abs(center_val - neighbor_val) >> 3;
                    if (diff > 31) diff = 31;

                    int w = spatial_w[kr + 2][kc + 2] * color_w_lut[diff];
                    
                    val_sum += neighbor_val * w;
                    weight_sum += w;
                }
            }
            
            img_out[r][c] = (pixel_t)(val_sum / weight_sum);
        }
    }
}

// ---------------------------------------------------------
// Golden Kernel 4: Sobel Gradients
// ---------------------------------------------------------
static void golden_kernel4(pixel_t stage1[HEIGHT][WIDTH], pixel_t stage2_x[HEIGHT][WIDTH], pixel_t stage2_y[HEIGHT][WIDTH]) {
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
// Golden Kernel 5: Magnitude Gradient Direction
// ---------------------------------------------------------
static void golden_kernel5(pixel_t stage2_x[HEIGHT][WIDTH], pixel_t stage2_y[HEIGHT][WIDTH], pixel_t stage3_mag[HEIGHT][WIDTH], pixel_t stage3_dir[HEIGHT][WIDTH]) {
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

// ---------------------------------------------------------
// Kernel 6: Non-Maximum Suppression
// ---------------------------------------------------------
static void golden_kernel6(pixel_t stage3_mag[HEIGHT][WIDTH], pixel_t stage3_dir[HEIGHT][WIDTH], pixel_t stage4[HEIGHT][WIDTH]) {
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
            } else if (dir == 90) {
                mag1 = stage3_mag[r - 1][c]; 
                mag2 = stage3_mag[r + 1][c];
            } else if (dir == 45) {
                mag1 = stage3_mag[r + 1][c - 1]; 
                mag2 = stage3_mag[r - 1][c + 1];
            } else { 
                mag1 = stage3_mag[r - 1][c - 1]; 
                mag2 = stage3_mag[r + 1][c + 1];
            }
            
            stage4[r][c] = (mag >= mag1 && mag >= mag2) ? mag : (pixel_t)0;
        }
    }
}

// ---------------------------------------------------------
// Golden Kernel 7: Hysteresis Thresholding
// ---------------------------------------------------------
static void golden_kernel7(pixel_t stage4[HEIGHT][WIDTH], pixel_t img_out[HEIGHT][WIDTH]) {
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
// Kernel 8: Morphological Dilation
// ---------------------------------------------------------
static void golden_kernel8(pixel_t stage5[HEIGHT][WIDTH], pixel_t img_out[HEIGHT][WIDTH]) {
    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
            
            if (r == 0 || r == HEIGHT - 1 || c == 0 || c == WIDTH - 1) {
                img_out[r][c] = 0;
                continue;
            }

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
// Golden Pipeline Integration (8-Stage Architecture)
// ---------------------------------------------------------
static void golden_vision_pipeline(
    pixel_t img_r[HEIGHT][WIDTH], 
    pixel_t img_g[HEIGHT][WIDTH], 
    pixel_t img_b[HEIGHT][WIDTH], 
    pixel_t img_out[HEIGHT][WIDTH]
) {
    static pixel_t stage0_gray[HEIGHT][WIDTH];
    static pixel_t stage1_median[HEIGHT][WIDTH];
    static pixel_t stage2_blur[HEIGHT][WIDTH];
    static pixel_t stage3_sobel_x[HEIGHT][WIDTH];
    static pixel_t stage3_sobel_y[HEIGHT][WIDTH];
    static pixel_t stage4_mag[HEIGHT][WIDTH];
    static pixel_t stage4_dir[HEIGHT][WIDTH];
    static pixel_t stage5_nms[HEIGHT][WIDTH];
    static pixel_t stage6_hyst[HEIGHT][WIDTH];

    golden_kernel1(img_r, img_g, img_b, stage0_gray);
    golden_kernel2(stage0_gray, stage1_median);
    golden_kernel3(stage1_median, stage2_blur);
    golden_kernel4(stage2_blur, stage3_sobel_x, stage3_sobel_y);
    golden_kernel5(stage3_sobel_x, stage3_sobel_y, stage4_mag, stage4_dir);
    golden_kernel6(stage4_mag, stage4_dir, stage5_nms);
    golden_kernel7(stage5_nms, stage6_hyst);
    golden_kernel8(stage6_hyst, img_out);
}


// ---------------------------------------------------------
// Main Testbench
// ---------------------------------------------------------
int main() {
    static pixel_t in_r[HEIGHT][WIDTH];
    static pixel_t in_g[HEIGHT][WIDTH];
    static pixel_t in_b[HEIGHT][WIDTH];
    static pixel_t out_hw[HEIGHT][WIDTH];
    static pixel_t out_gold[HEIGHT][WIDTH];

    init_input_rgb(in_r, in_g, in_b);

    std::cout << "Running Hardware Design...\n";
    top_kernel(in_r, in_g, in_b, out_hw);

    std::cout << "Running Golden Reference...\n";
    golden_vision_pipeline(in_r, in_g, in_b, out_gold);

    // Verify Correctness with +/- 1% Tolerance
    int errors = 0;
    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
            
            // Convert ap_fixed to double for testbench math
            double hw_val = out_hw[r][c].to_double();
            double gold_val = out_gold[r][c].to_double();
            
            // Calculate absolute difference
            double diff = std::abs(hw_val - gold_val);
            
            // Calculate the 1% allowed error threshold
            double allowed_error = 0.01 * std::abs(gold_val);

            // Check if the difference exceeds the 1% allowed error
            if (diff > allowed_error) {
                errors++;
                std::cout << "Mismatch at [" << r << "][" << c << "]"
                            << " hw=" << hw_val
                            << " gold=" << gold_val
                            << " (diff=" << diff << " > allowed=" << allowed_error << ")\n";
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
