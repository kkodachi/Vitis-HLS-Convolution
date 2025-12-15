#include "config.h"

// PARAMETERS FOR conv3d()
#define MAX_CONV_H 224 // max height of input to conv kernel (conv1)
#define MAX_CONV_W 224 // max width of input to conv kernel (conv1)
#define MAX_CONV1_IC 3 // max input channels to conv1 kernel (conv1)
#define MAX_CONV1_OC 96 // max output channels to conv kernel (conv1)

#define MAX_CONV10_H 14 // max height of input to conv kernel (conv1)
#define MAX_CONV10_W 14 // max width of input to conv kernel (conv1)

#define MAX_CONV10_IC 512 // max input channels to conv kernel (conv10)

#define MAX_CONV_K 7 // max kernel size for conv kernel (conv1)

// parameters for fire()
// #define MAX_FIRE_H 56 // max height of input to fire module
// #define MAX_FIRE_W 56 // max width of input to fire module
#define MAX_FIRE_H 112
#define MAX_FIRE_W 112
#define MAX_FIRE_IC 512 // max input channels
#define MAX_FIRE_SC 64 // max squeeze channels
#define MAX_FIRE_EC 256 // max expand channels

// parameters for avgpool()
#define AVGPOOL_H 14
#define AVGPOOL_W 14
#define AVGPOOL_C 10
// #define NUM_CLASSES 10

// parameters for top module and controller
#define MODULES 4
#define CONV_IND 0
#define MAXPOOL_IND 1
#define FIRE_IND 2
#define AVGPOOL_IND 3

void conv3d(
    bool enable,
    fixed_point_t activations[MAX_CONV_H][MAX_CONV_W][MAX_CONV1_IC],
    fixed_point_t weights[MAX_CONV_K][MAX_CONV_K][MAX_CONV1_IC][MAX_CONV1_OC],
    // fixed_point_t output[MAX_CONV_H][MAX_CONV_W][MAX_CONV_OC],
    fixed_point_t output[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],

    int H,      // input height
    int W,      // input width
    int IC,     // input channels
    int OC,     // output channels
    int K,      // kernel size
    int S,      // stride
    int P       // padding
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
    const fixed_point_t activations[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],
    fixed_point_t output[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],
    int H,      // input height
    int W,      // input width
    int IC     // input channels
);

void avgpool(
    bool enable,
    const fixed_point_t activations[AVGPOOL_H][AVGPOOL_W][AVGPOOL_C],
    fixed_point_t output[AVGPOOL_C],
    int H,      // input height
    int W,      // input width
    int IC     // input channels
);

struct Args {
    int IH;
    int IW;
    int IC;
    // padding, stride, input/output/weight arrays etc
    // just have a datamember for every necessary parameter across all layers and only use necessary ones for each layer
    // for example avgpool doesn't need much so just ignore the other parameters

    // Args(int ih, int iw, int ic)
    //     : IH(ih), IW(iw), IC(ic) {}
    
};

void controller(
    bool enables[],
    int layer,
    Args layers[]
);

void conv1(
    bool enable,
    fixed_point_t activations[MAX_CONV_H][MAX_CONV_W][MAX_CONV1_IC],
    fixed_point_t weights[MAX_CONV_K][MAX_CONV_K][MAX_CONV1_IC][MAX_CONV1_OC],
    // fixed_point_t output[MAX_CONV_H][MAX_CONV_W][MAX_CONV_OC],
    fixed_point_t output[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],

    int H,      // input height
    int W,      // input width
    int IC,     // input channels
    int OC     // output channels
);

void conv10(
    bool enable,
    fixed_point_t activations[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],
    fixed_point_t weights[MAX_FIRE_IC][AVGPOOL_C],
    fixed_point_t output[AVGPOOL_H][AVGPOOL_W][AVGPOOL_C]
);