#include "config.h"
// #include <stddef.h>


#define MAX_H   32      // largest CIFAR-10 input height
#define MAX_W   32      // largest CIFAR-10 input width

#define MAX_IC  512     // max input channels
#define MAX_OC  512     // max output channels

#define MAX_K   3       // largest kernel in squeezenet

void conv3d_ws(
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
)