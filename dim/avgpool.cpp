#include "kernel.h"
#include "config.h"

void avgpool(
    bool enable,
    const fixed_point_t activations[AVGPOOL_H][AVGPOOL_W][AVGPOOL_C],
    fixed_point_t output[AVGPOOL_C],
    int H,      // input height
    int W,      // input width
    int IC     // input channels
)
{
    if (!enable) return;
    
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
        output[ic] = sum / (H * W);
    }
}