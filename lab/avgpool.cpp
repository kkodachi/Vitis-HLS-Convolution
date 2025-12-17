#include "config.h"
#include "kernel.h"

void avgpool(
    const fixed_point_t activations[CONV5_DIM][CONV5_DIM][NUM_CLASSES],
    fixed_point_t output[NUM_CLASSES],
    int H,      // input height
    int W,      // input width
    int IC     // input channels
)
{
    
    IC_LOOP:
    for (int ic = 0; ic < IC; ic++) {
        accum_t sum = 0;

        KH_LOOP:
        for (int h = 0; h < H; h++) {
            KW_LOOP:
            for (int w = 0; w < W; w++) {
                #pragma HLS PIPELINE II=1
                sum += activations[h][w][ic];
            }
        }
        // output[ic] = (fixed_point_t)(sum / (H * W));
        accum_t avg = sum / (H * W);
        if (avg < -128) avg = -128;
        else if (avg > 127) avg = 127;
        output[ic] = (fixed_point_t)avg;
    }
}