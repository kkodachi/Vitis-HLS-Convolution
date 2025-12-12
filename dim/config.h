// #include "ap_fixed.h"
// #include <stdint.h>

// fixed-point type: 4 integer bits, 4 fractional bits
typedef ap_fixed<8, 4, AP_RND, AP_SAT> fixed_point_t;
// typedef int fixed_point_t;

typedef ap_fixed<24, 12, AP_RND, AP_SAT> accum_t;
// typedef int accum_t;
