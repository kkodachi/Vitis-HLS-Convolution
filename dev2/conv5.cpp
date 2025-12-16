#include "config.h"
#include "kernel.h"

void conv5(
    fixed_point_t activations[MAX_CONV_DIM][MAX_CONV_DIM][MAX_CONV_DIM],
    fixed_point_t weights[MAX_CONV_DIM][NUM_CLASSES],
    // fixed_point_t output[MAX_CONV_H][MAX_CONV_W][MAX_CONV_OC],
    fixed_point_t output[CONV5_DIM][CONV5_DIM][NUM_CLASSES],

    int H,      // input height
    int W,      // input width
    int IC,     // input channels,
    int OC
)
{
    const int K = 1;    // kernel size
    const int S = 1;    // stride
    const int P = 0;    // padding
    OC = 10;     // output channels
    int H_OUT = (H + 2*P - K)/S + 1; // 14
    int W_OUT = (W + 2*P - K)/S + 1; // 14

    OC_LOOP:
    for (int oc = 0; oc < OC; oc++) {
        H_LOOP:
        for (int h = 0; h < H; h++) {
            W_LOOP:
            for (int w = 0; w < W; w++) {
                accum_t psum = 0;
                IC_LOOP:
                for (int ic = 0; ic < IC; ic++) {
                    #pragma HLS PIPELINE II=1
                    psum += activations[h][w][ic] * weights[ic][oc];
                }
                // output[h][w][oc] = (fixed_point_t)psum;
                if (psum < -128) psum = -128;
                else if (psum > 127) psum = 127;
                output[h][w][oc] = (fixed_point_t)psum;
            }
        }
    }
}