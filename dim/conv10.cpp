#include "config.h"
#include "kernel.h"

void conv10(
    bool enable,
    fixed_point_t activations[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],
    fixed_point_t weights[MAX_FIRE_IC][AVGPOOL_C],
    // fixed_point_t output[MAX_CONV_H][MAX_CONV_W][MAX_CONV_OC],
    fixed_point_t output[AVGPOOL_H][AVGPOOL_W][AVGPOOL_C],

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
    int H_OUT = (H + 2*P - K)/S + 1; // 56
    int W_OUT = (W + 2*P - K)/S + 1; // 56

    fixed_point_t input_local[MAX_CONV10_IC];
    // #pragma HLS ARRAY_PARTITION variable=input_local cyclic factor=4
    #pragma HLS bind_storage variable=input_local type=ram_2p impl=bram

    #pragma HLS ARRAY_PARTITION variable=weights cyclic factor=4 dim=1
//	#pragma HLS bind_storage variable=local_activations type=ram_2p impl=bram

    H_LOOP:
    for (int h = 0; h < H; h++) {
        W_LOOP:
        for (int w = 0; w < W; w++) {
            LOAD_INPUT:
            for (int ic = 0; ic < IC; ic++) {
                #pragma HLS PIPELINE II=1
                input_local[ic] = activations[h][w][ic];
            }

            OC_LOOP:
            for (int oc = 0; oc < OC; oc++) {
                accum_t sum = 0;

                IC_LOOP:
                for (int ic = 0; ic < IC; ic++) {
                    #pragma HLS PIPELINE II=1
                    sum += input_local[ic] * weights[ic][oc];
                }

                output[h][w][oc] = (fixed_point_t)sum;
            }
        }
    }
}