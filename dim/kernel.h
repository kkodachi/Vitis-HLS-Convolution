#include "config.h"
// #include <stddef.h>


#define MAX_H   32      // largest CIFAR-10 input height
#define MAX_W   32      // largest CIFAR-10 input width

#define MAX_IC  512     // max input channels
#define MAX_OC  512     // max output channels

#define MAX_K   3       // largest kernel in squeezenet

// PARAMETERS FOR conv3d()
#define MAX_CONV_H 224 // max height of input to conv kernel (conv1)
#define MAX_CONV_W 224 // max width of input to conv kernel (conv1)
#define MAX_CONV_IC 512 // max input channels to conv kernel (conv10)
#define MAX_CONV_OC 96 // max output channels to conv kernel (conv1)
#define MAX_CONV_K 7 // max kernel size for conv kernel (conv1)

// parameters for fire()
#define MAX_FIRE_H 56 // max height of input to fire module
#define MAX_FIRE_W 56 // max width of input to fire module
#define MAX_FIRE_IC 512 // max input channels
#define MAX_FIRE_SC 64 // max squeeze channels
#define MAX_FIRE_EC 256 // max expand channels

// TODO: add top function to call entire model

void conv3d(
    bool enable,
    fixed_point_t activations[MAX_H * MAX_W * MAX_IC],
    fixed_point_t weights[MAX_K * MAX_K * MAX_IC * MAX_OC],
    fixed_point_t output[MAX_H * MAX_W * MAX_OC],

    int H,      // input height
    int W,      // input width
    int IC,     // input channels
    int OC,     // output channels
    int K,      // kernel size
    int S, // stride
    int P     // padding
);

void fire(
    bool enable,
    const fixed_point_t input[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],
    const fixed_point_t squeeze_weights[MAX_FIRE_IC][MAX_FIRE_SC],
    const fixed_point_t expand1x1_weights[MAX_FIRE_SC][MAX_FIRE_EC],
    const fixed_point_t expand3x3_weights[3][3][MAX_FIRE_SC][MAX_FIRE_EC],
    fixed_point_t output[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],
    int H,
    int W,
    int IC,
    int SC, // squeeze channels
    int EC // expand channels
);

void maxpool(
    bool enable,
    const fixed_point_t activations[MAX_H * MAX_W * MAX_IC],
    fixed_point_t output[MAX_H * MAX_W * MAX_IC],
    int H,      // input height
    int W,      // input width
    int IC,     // input channels
    int K,      // kernel size
    int S  // stride
);

void avgpool(
    bool enable,
    const fixed_point_t activations[MAX_H * MAX_W * MAX_IC],
    fixed_point_t output[MAX_H * MAX_W * MAX_IC],
    int H,      // input height
    int W,      // input width
    int IC,     // input channels
    int K,      // kernel size
    int stride  // stride
);