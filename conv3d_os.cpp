#include "config.h"
#include "conv3d_kernel.h"

void conv3d_os(
    data_type *activations,
    size_t H_in, size_t W_in, size_t D_in,size_t C_in,
    data_type *weights,
    size_t Kh, size_t Kw, size_t Kd, size_t C_out,
    data_type *output,
    size_t stride_d = 1, size_t stride_h = 1, size_t stride_w = 1,
    size_t pad_d = 1, size_t pad_h = 1, size_t pad_w = 1
)
{

}