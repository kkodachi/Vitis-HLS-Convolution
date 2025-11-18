#include "config.h"
#include "conv3d_kernel.h"


void conv3d_ws(
    data_type *activations,
    size_t H_in, size_t W_in, size_t D_in,size_t C_in,
    data_type *weights,
    size_t Kh, size_t Kw, size_t Kd, size_t C_out,
    data_type *output,
    size_t stride_d = 1, size_t stride_h = 1, size_t stride_w = 1,
    size_t pad_d = 1, size_t pad_h = 1, size_t pad_w = 1
)
{
    data_type A_BRAM[BUFFER_SIZE];
    data_type W_BRAM[BUFFER_SIZE];

    // output dimensions
    size_t D_out = (D_in + 2*pad_d - Kd) / stride_d + 1;
    size_t H_out = (H_in + 2*pad_h - Kh) / stride_h + 1;
    size_t W_out = (W_in + 2*pad_w - Kw) / stride_w + 1;
    // number of weights
    const size_t W_count = C_out * C_in * Kd * Kh * Kw;

    // weights for one channel
    const size_t WEIGHTS_PER_OC = C_in * Kd * Kh * Kw;
    // check if valid tile exists
    size_t max_tile_oc = BUFFER_SIZE / WEIGHTS_PER_OC;
    // return if can't fit 1 channel into BRAM
    if (max_tile_oc == 0) {
        return;
    }

    // conv operation over each channel using the tiles

}