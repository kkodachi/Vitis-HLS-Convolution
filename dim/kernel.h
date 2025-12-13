#ifndef KERNEL_H
#define KERNEL_H

#include "config.h"

// PARAMETERS FOR conv3d()
#define MAX_CONV_H 224 // max height of input to conv kernel (conv1)
#define MAX_CONV_W 224 // max width of input to conv kernel (conv1)
#define MAX_CONV_IC 512 // max input channels to conv kernel (conv10)
#define MAX_CONV_OC 96 // max output channels to conv kernel (conv1)
#define MAX_CONV_K 7 // max kernel size for conv kernel (conv1)

// parameters for fire()
#define MAX_FIRE_H 112
#define MAX_FIRE_W 112
#define MAX_FIRE_IC 512 // max input channels
#define MAX_FIRE_SC 64 // max squeeze channels
#define MAX_FIRE_EC 256 // max expand channels

// parameters for avgpool()
#define AVGPOOL_H 14
#define AVGPOOL_W 14
#define AVGPOOL_C 10
#define NUM_CLASSES 10

// parameters for top module and controller
#define MODULES 4
#define CONV_IND 0
#define MAXPOOL_IND 1
#define FIRE_IND 2
#define AVGPOOL_IND 3

#define TOTAL_LAYERS 14

void conv3d(
    bool enable,
    fixed_point_t activations[MAX_CONV_H][MAX_CONV_W][MAX_CONV_IC],
    fixed_point_t weights[MAX_CONV_K][MAX_CONV_K][MAX_CONV_IC][MAX_CONV_OC],
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
    // Enable signals for each module type
    bool enable_conv;
    bool enable_maxpool;
    bool enable_fire;
    bool enable_avgpool;
    
    // Input/output buffer pointers
    fixed_point_t (*input_buf)[MAX_FIRE_W][MAX_FIRE_IC];
    fixed_point_t (*output_buf)[MAX_FIRE_W][MAX_FIRE_IC];
    
    // For avgpool (has different dimensions)
    fixed_point_t (*avgpool_input)[AVGPOOL_W][AVGPOOL_C];
    fixed_point_t *avgpool_output;
    
    // Conv parameters
    fixed_point_t (*conv_weights)[MAX_CONV_K][MAX_CONV_IC][MAX_CONV_OC];
    int H;       // input height
    int W;       // input width
    int IC;      // input channels
    int OC;      // output channels (for conv)
    int K;       // kernel size (for conv)
    int S;       // stride (for conv)
    int P;       // padding (for conv)
    
    // Fire module parameters
    fixed_point_t (*squeeze_weights)[MAX_FIRE_SC];
    fixed_point_t (*expand1x1_weights)[MAX_FIRE_EC];
    fixed_point_t (*expand3x3_weights)[3][MAX_FIRE_SC][MAX_FIRE_EC];
    int SC;      // squeeze channels
    int EC;      // expand channels
    
    // Constructor to initialize all pointers to nullptr
    Args() : enable_conv(false), enable_maxpool(false), enable_fire(false), enable_avgpool(false),
             input_buf(nullptr), output_buf(nullptr), 
             avgpool_input(nullptr), avgpool_output(nullptr),
             conv_weights(nullptr), H(0), W(0), IC(0), OC(0), K(0), S(0), P(0),
             squeeze_weights(nullptr), expand1x1_weights(nullptr), expand3x3_weights(nullptr),
             SC(0), EC(0) {}
};

void controller(
    int layer,
    Args &args
);

#endif // KERNEL_H