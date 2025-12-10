#include "config.h"
// #include <stddef.h>


#define MAX_H   32      // largest CIFAR-10 input height
#define MAX_W   32      // largest CIFAR-10 input width

#define MAX_IC  512     // max input channels
#define MAX_OC  512     // max output channels

#define MAX_K   3       // largest kernel in squeezenet

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
    int stride, // stride
    int pad     // padding
);

void maxpool(
    bool enable,
    const fixed_point_t activations[MAX_H * MAX_W * MAX_IC],
    fixed_point_t output[MAX_H * MAX_W * MAX_IC],
    int H,      // input height
    int W,      // input width
    int IC,     // input channels
    int K,      // kernel size
    int stride  // stride
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

void relu(
    bool enable,
    fixed_point_t activations[MAX_H * MAX_W * MAX_IC],
    int H,      // input height
    int W,      // input width
    int IC     // input channels
);

void fire_module(
    bool enable,
    const fixed_point_t input[MAX_H * MAX_W * MAX_IC],
    const fixed_point_t squeeze_weights[1 * 1 * MAX_IC * MAX_OC],
    const fixed_point_t expand1x1_weights[1 * 1 * MAX_IC * MAX_OC],
    const fixed_point_t expand3x3_weights[MAX_K * MAX_K * MAX_IC * MAX_OC],
    fixed_point_t output[MAX_H * MAX_W * MAX_OC],
    int H,
    int W,
    int IC,
    int squeeze_ch,
    int expand_ch
);

void controller(
    int stage,
    bool* en_conv,
    bool* en_maxpool,
    bool* en_avgpool,
    bool* en_relu,
    bool* en_fire
);

void squeezenet_top(
    fixed_point_t input[MAX_H * MAX_W * MAX_IC],
    fixed_point_t output[10],
    fixed_point_t* all_weights,
    int num_stages
);