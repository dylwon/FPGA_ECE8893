#ifndef DCL_H
#define DCL_H

#include <ap_fixed.h>

#define HEIGHT 256
#define WIDTH 256

// Fixed-point data types satisfying the rubric requirement
typedef ap_fixed<16, 8, AP_TRN, AP_WRAP> pixel_t;
typedef ap_fixed<32, 16, AP_TRN, AP_WRAP> calc_t;
typedef ap_fixed<16, 2, AP_TRN, AP_WRAP> coef_t;

// Prototype for your hardware design
void top_kernel(pixel_t img_in[HEIGHT][WIDTH], pixel_t img_out[HEIGHT][WIDTH]);

#endif
