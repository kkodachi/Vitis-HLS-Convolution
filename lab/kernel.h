#include "config.h"

#define MAX_CONV_DIM 256
#define MAX_CONV_K 3

#define CONV1_DIM 224
#define CONV1_IC 3
#define CONV1_OC 32

#define CONV2_DIM 112
#define CONV2_IC 32
#define CONV2_OC 64

#define CONV3_DIM 56
#define CONV3_IC 64
#define CONV3_OC 128

#define CONV4_DIM 28
#define CONV4_IC 28
#define CONV4_OC 256

#define CONV5_DIM 14
#define CONV5_IC 256
// #define CONV5_OC 10

// AVGPOOL input CONV5_DIM
#define NUM_CLASSES 10

void squeezenet(
    fixed_point_t input[CONV1_DIM * CONV1_DIM * CONV1_IC],
    fixed_point_t* output,
    fixed_point_t conv1w[MAX_CONV_K * MAX_CONV_K * CONV1_IC * CONV1_OC],
    fixed_point_t conv2w[MAX_CONV_K * MAX_CONV_K * CONV2_IC *CONV2_OC],
    fixed_point_t conv3w[MAX_CONV_K * MAX_CONV_K * CONV3_IC *CONV3_OC],
    fixed_point_t conv4w[MAX_CONV_K * MAX_CONV_K * CONV4_IC *CONV4_OC],
    fixed_point_t conv5w[CONV4_OC * NUM_CLASSES]
);

void conv(
    fixed_point_t activations[MAX_CONV_DIM][MAX_CONV_DIM][MAX_CONV_DIM],
    fixed_point_t weights[MAX_CONV_K][MAX_CONV_K][MAX_CONV_DIM][MAX_CONV_DIM],
    fixed_point_t output[MAX_CONV_DIM][MAX_CONV_DIM][MAX_CONV_DIM],
    int H,
    int W,
    int IC,
    int OC
);

void conv5(
    fixed_point_t activations[MAX_CONV_DIM][MAX_CONV_DIM][MAX_CONV_DIM],
    fixed_point_t weights[MAX_CONV_DIM][NUM_CLASSES],
    // fixed_point_t output[MAX_CONV_H][MAX_CONV_W][MAX_CONV_OC],
    fixed_point_t output[CONV5_DIM][CONV5_DIM][NUM_CLASSES],

    int H,      // input height
    int W,      // input width
    int IC,     // input channels,
    int OC
);

void avgpool(
    const fixed_point_t activations[CONV5_DIM][CONV5_DIM][NUM_CLASSES],
    fixed_point_t output[NUM_CLASSES],
    int H,      // input height
    int W,      // input width
    int IC     // input channels
);
