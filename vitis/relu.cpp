
#include "kernel.h"

void relu(
    fixed_point_t activations[MAX_H * MAX_W * MAX_IC],
    int H,      // input height
    int W,      // input width
    int IC     // input channels
)
{
    H_LOOP:
    for (int h = 0; h < H; h++){
        W_LOOP:
        for (int w = 0; w < W; w++){
            CH_LOOP:
            for (int c = 0; c < IC; c++){
                #pragma HLS PIPELINE II=1
                int idx = h * (MAX_W * MAX_IC) + w * (MAX_IC) + c;
                fixed_point_t x = activations[idx];
                activations[idx] = (x > 0) ? x : (fixed_point_t)0;
            }
        }
    }
}