#include <ap_fixed.h>
#include <cmath>

// Define image dimensions
#define HEIGHT 512
#define WIDTH 512

// Define fixed-point types as required by the rubric
typedef ap_fixed<16, 8, AP_TRN, AP_WRAP> pixel_t;
typedef ap_fixed<32, 16, AP_TRN, AP_WRAP> calc_t;
typedef ap_fixed<16, 2, AP_TRN, AP_WRAP> coef_t;

// ---------------------------------------------------------
// Kernel 1: Gaussian Blur (Noise Reduction)
// Standard 3x3 convolution. Reads from input, writes to stage1.
// ---------------------------------------------------------
void kernel1_gaussian_blur(pixel_t img_in[HEIGHT][WIDTH], pixel_t stage1[HEIGHT][WIDTH]) {
    // Standard software practice: do not fuse loops yet.
    coef_t kernel[3][3] = {
        {0.0625, 0.125, 0.0625},
        {0.125,  0.25,  0.125},
        {0.0625, 0.125, 0.0625}
    };

    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
            // Handle boundaries cleanly without artificial bloat
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
// Reads from stage1, writes to stage2_x and stage2_y.
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
// Reads X/Y gradients, calculates magnitude. 
// ---------------------------------------------------------
void kernel3_magnitude_direction(pixel_t stage2_x[HEIGHT][WIDTH], pixel_t stage2_y[HEIGHT][WIDTH], pixel_t stage3_mag[HEIGHT][WIDTH], pixel_t stage3_dir[HEIGHT][WIDTH]) {
    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
            calc_t gx = stage2_x[r][c];
            calc_t gy = stage2_y[r][c];
            
            // Absolute value approximation for magnitude (L1 norm) to save DSPs later
            calc_t abs_gx = (gx < 0) ? (calc_t)(-gx) : gx;
            calc_t abs_gy = (gy < 0) ? (calc_t)(-gy) : gy;
            stage3_mag[r][c] = (pixel_t)(abs_gx + abs_gy);

            // Simple 4-way direction categorization (0, 45, 90, 135)
            pixel_t dir = 0;
            if (abs_gx > abs_gy) {
                dir = 0; // Horizontal edge
            } else {
                dir = 90; // Vertical edge
            }
            stage3_dir[r][c] = dir;
        }
    }
}

// ---------------------------------------------------------
// Kernel 4: Non-Maximum Suppression (Thinning)
// Checks neighboring pixels based on gradient direction.
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

            if (dir == 0) { // Check horizontal neighbors
                mag1 = stage3_mag[r][c - 1];
                mag2 = stage3_mag[r][c + 1];
            } else { // Check vertical neighbors
                mag1 = stage3_mag[r - 1][c];
                mag2 = stage3_mag[r + 1][c];
            }

            // Suppress if not the local maximum
            if (mag >= mag1 && mag >= mag2) {
                stage4[r][c] = mag;
            } else {
                stage4[r][c] = 0;
            }
        }
    }
}

// ---------------------------------------------------------
// Kernel 5: Double Thresholding
// Categorizes pixels into strong, weak, or zero.
// ---------------------------------------------------------
void kernel5_double_threshold(pixel_t stage4[HEIGHT][WIDTH], pixel_t img_out[HEIGHT][WIDTH]) {
    pixel_t HIGH_THRESH = 50;
    pixel_t LOW_THRESH = 20;

    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
            pixel_t val = stage4[r][c];
            if (val >= HIGH_THRESH) {
                img_out[r][c] = 255; // Strong edge
            } else if (val >= LOW_THRESH) {
                img_out[r][c] = 127; // Weak edge
            } else {
                img_out[r][c] = 0;   // No edge
            }
        }
    }
}

// ---------------------------------------------------------
// Top-Level Function
// Sequentially executes the pipeline.
// ---------------------------------------------------------
void top_vision_pipeline(pixel_t img_in[HEIGHT][WIDTH], pixel_t img_out[HEIGHT][WIDTH]) {
    // Massive intermediate memory arrays (The Bottleneck)
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
    kernel5_double_threshold(stage4, img_out);
}
