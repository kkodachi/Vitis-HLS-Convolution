#include "ap_fixed.h"
// #include <stdint.h>

// fixed-point type: 4 integer bits, 4 fractional bits
// typedef ap_fixed<8, 4, AP_RND, AP_SAT> fixed_point_t;
typedef ap_fixed<16, 6, AP_RND, AP_SAT> fixed_point_t;

// Accumulator type: 12 integer bits, 12 fractional bits
// typedef ap_fixed<16, 12> accum_t;

// typedef fixed_point_t act_t;
// typedef fixed_point_t weight_t;
// typedef ap_fixed<24, 12, AP_RND, AP_SAT> accum_t;
typedef ap_fixed<48, 24, AP_RND, AP_SAT> acc_t; // for large K*IC
