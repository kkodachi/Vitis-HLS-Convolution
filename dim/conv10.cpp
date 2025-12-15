#include "config.h"
#include "kernel.h"


void conv10(
    bool enable,
    fixed_point_t activations[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],
    fixed_point_t weights[MAX_FIRE_IC][AVGPOOL_C],
    // fixed_point_t output[MAX_CONV_H][MAX_CONV_W][MAX_CONV_OC],
    fixed_point_t output[AVGPOOL_H][AVGPOOL_W][AVGPOOL_C]

//    int H,      // input height
//    int W,      // input width
//    int IC,     // input channels,
//    int OC
)
{
	if (!enable) return;

	OC_LOOP:
    for (int oc = 0; oc < AVGPOOL_C; oc++) {
        H_LOOP:
        for (int h = 0; h < AVGPOOL_H; h++) {
            W_LOOP:
            for (int w = 0; w < AVGPOOL_W; w++) {
                accum_t psum = 0;
                IC_LOOP:
                for (int ic = 0; ic < MAX_FIRE_IC; ic++) {
                    #pragma HLS PIPELINE II=1
                    psum += activations[h][w][ic] * weights[ic][oc];
                }
                output[h][w][oc] = (psum > 0) ? (fixed_point_t)psum : (fixed_point_t)0;
            }
        }
    }
}

// void conv10(
//    bool enable,
//    fixed_point_t activations[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],
//    fixed_point_t weights[MAX_FIRE_IC][AVGPOOL_C],
//    // fixed_point_t output[MAX_CONV_H][MAX_CONV_W][MAX_CONV_OC],
//    fixed_point_t output[AVGPOOL_H][AVGPOOL_W][AVGPOOL_C],

//    int H,      // input height
//    int W,      // input width
//    int IC,     // input channels,
//    int OC
// )
// {
//     const int K = 1;    // kernel size
//     const int S = 1;    // stride
//     const int P = 0;    // padding
//     int H_OUT = (H + 2*P - K)/S + 1; // 14
//     int W_OUT = (W + 2*P - K)/S + 1; // 14

//     fixed_point_t activations_local[AVGPOOL_H][AVGPOOL_W];

//     fixed_point_t output_local[AVGPOOL_H][AVGPOOL_W];

//     IC_LOOP:
//     for (int ic = 0; ic < MAX_FIRE_IC; ic++) {
//         LOAD_A:
//         for (int h = 0; h < AVGPOOL_H; h++){
//             for (int w = 0; w < AVGPOOL_W; w++){
//                 #pragma HLS PIPELINE II=1
//                 activations_local[h][w] = activations[h][w][ic]
//                 output_local[h][w] = 0;
//             }
//         }

//         for (int oc = 0; oc < AVGPOOL_C; oc++){
//             #pragma HLS PIPELINE II=1
//             fixed_point_t w = weights[ic][oc]
//             H_LOOP:
//             for (int h = 0; h < AVGPOOL_H; h++) {
                
//                 W_LOOP:
//                 for (int w = 0; w < AVGPOOL_W; w++) {
//                     #pragma HLS UNROLL
                    
//                 }
//             }
//         }
//     }
// }
