// #include "ap_fixed.h"
// #include <stdint.h>

// fixed-point type: 4 integer bits, 4 fractional bits
typedef float fixed_point_t;
// typedef ap_fixed<16, 6, AP_RND, AP_SAT> fixed_point_t;

// Accumulator type: 12 integer bits, 12 fractional bits
// typedef ap_fixed<16, 12> accum_t;

// typedef fixed_point_t act_t;
// typedef fixed_point_t weight_t;
typedef float accum_t;
// typedef ap_fixed<48, 24, AP_RND, AP_SAT> accum_t; // for large K*IC
