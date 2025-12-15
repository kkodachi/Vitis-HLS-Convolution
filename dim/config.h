// #include "ap_fixed.h"
// #include <stdint.h>

// fixed-point type: 4 integer bits, 4 fractional bits
//typedef ap_fixed<8, 4, AP_RND, AP_SAT> fixed_point_t;
typedef int fixed_point_t;

//typedef ap_fixed<24, 12, AP_RND, AP_SAT> accum_t;
typedef int accum_t;

// #ifndef CONFIG_H
// #define CONFIG_H

// // Detect if we're in Vitis HLS or regular C++ environment
// #ifdef __SYNTHESIS__
//     // Running in Vitis HLS - use real ap_fixed
//     #include "ap_fixed.h"
// #else
//     // Running on regular system (MacBook/Linux) - use mock
//     #ifndef AP_FIXED_MOCK_H
//         #include "ap_fixed_mock.h"
//     #endif
// #endif

// // fixed-point type: 8 bits total, 4 integer bits, 4 fractional bits
// typedef ap_fixed<8, 4, AP_RND, AP_SAT> fixed_point_t;

// // accumulator type: wider precision for intermediate calculations
// typedef ap_fixed<24, 12, AP_RND, AP_SAT> accum_t;

// #endif // CONFIG_H