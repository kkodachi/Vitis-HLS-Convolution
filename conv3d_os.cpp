#include "config.h"
#include "conv3d_kernel.h"

void conv2d_ws(
    fixed_point_t activations[MAX_H][MAX_W][MAX_IC],
    fixed_point_t weights[MAX_K][MAX_K][MAX_IC][MAX_OC],
    fixed_point_t output[MAX_H][MAX_W][MAX_OC],

    int H,      // input height
    int W,      // input width
    int IC,     // input channels
    int OC,     // output channels
    int K,      // kernel size
    int stride, // stride
    int pad     // padding
);
