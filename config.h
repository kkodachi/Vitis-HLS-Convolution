#include "ap_fixed.h"
// #include <stdint.h>

// fixed-point type: 4 integer bits, 4 fractional bits
typedef ap_fixed<8, 4, AP_RND, AP_SAT> fixed_point_t;

// Accumulator type: 12 integer bits, 4 fractional bits
typedef ap_fixed<16, 12> accum_t;
