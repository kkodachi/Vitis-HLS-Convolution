#include "kernel.h"

void relu(
    bool enable,
    fixed_point_t activations[MAX_H * MAX_W * MAX_IC],
    int H,      // input height
    int W,      // input width
    int IC     // input channels
)
{
    if (!enable) return;

    #pragma HLS INLINE off

    #pragma HLS INTERFACE s_axilite port=enable
    #pragma HLS INTERFACE s_axilite port=return

    H_LOOP:
    for (int h = 0; h < H; h++){
        #pragma HLS LOOP_TRIPCOUNT min=1 max=32
        
        W_LOOP:
        for (int w = 0; w < W; w++){
            #pragma HLS LOOP_TRIPCOUNT min=1 max=32
            
            CH_LOOP:
            for (int c = 0; c < IC; c++){
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=3 max=512
                
                int idx = h * (MAX_W * MAX_IC) + w * (MAX_IC) + c;
                fixed_point_t x = activations[idx];
                activations[idx] = (x > 0) ? x : (fixed_point_t)0;
            }
        }
    }
}