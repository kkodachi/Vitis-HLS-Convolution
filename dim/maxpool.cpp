#include "kernel.h"
#include "config.h"

void maxpool(
    bool enable,
    const fixed_point_t activations[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],
    fixed_point_t output[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],
    int H,      // input height
    int W,      // input width
    int IC     // input channels
)
{
    if (!enable) return;
    fixed_point_t output_local[MAX_FIRE_H][MAX_FIRE_W];

    const int K = 3;
    const int S = 2;
    const int P = 0;

    // int H_OUT = (H + 2*P - K)/S + 1;
    // int W_OUT = (W + 2*P - K)/S + 1;

    int H_OUT = ((H + 2*P - K + S - 1) / S) + 1;  // Ceiling division trick
    int W_OUT = ((W + 2*P - K + S - 1) / S) + 1;  // Ceiling division trick


    IC_LOOP:
    for (int ic = 0; ic < IC;ic++){
        OH_LOOP:
        for (int oh = 0; oh < H_OUT; oh++){
            OW_LOOP:
            for (int ow = 0; ow < W_OUT; ow++){
                #pragma HLS PIPELINE II=1
                fixed_point_t max_val = activations[oh*S][ow*S][ic];
                KH_LOOP:
                for (int kh = 0; kh < K; kh++) {
                    #pragma HLS UNROLL
                    KW_LOOP:
                    for (int kw = 0; kw < K; kw++) {
                        #pragma HLS UNROLL

                        int ih = oh * S + kh;
                        int iw = ow * S + kw;

                        fixed_point_t v = activations[ih][iw][ic];
                        if (v > max_val)
                            max_val = v;
                    }
                }
                output_local[oh][ow] = max_val;
            }
        }
        for (int oh = 0; oh < H_OUT; oh++){
            for (int ow = 0; ow < W_OUT; ow++){
                #pragma HLS PIPELINE II=1
                output[oh][ow][ic] = output_local[oh][ow];
            }
        }
    }
}