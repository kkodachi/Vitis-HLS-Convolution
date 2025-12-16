#ifndef KERNEL_H
#define KERNEL_H

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
#define MAX_FIRE_H 112
#define MAX_FIRE_W 112
#define MAX_FIRE_IC 512 // max input channels
#define MAX_FIRE_SC 64 // max squeeze channels
#define MAX_FIRE_EC 256 // max expand channels

// parameters for avgpool()
#define AVGPOOL_H 14
#define AVGPOOL_W 14
#define AVGPOOL_C 10

// Total layers in SqueezeNet
#define TOTAL_LAYERS 14

// ============================================================================
// Layer Types
// ============================================================================
enum LayerType {
    LAYER_NONE,
    LAYER_CONV,      // Conv1
    LAYER_MAXPOOL,   // MaxPool layers
    LAYER_FIRE,      // Fire modules
    LAYER_CONV10,    // Final 1x1 conv
    LAYER_AVGPOOL    // Final avgpool
};

// ============================================================================
// Layer Configuration Structure
// ============================================================================
struct LayerConfig {
    LayerType layer_type;
    
    // Dimensions
    int H;
    int W;
    int IC;
    int OC;
    
    // Convolution parameters
    int K;  // kernel size
    int S;  // stride
    int P;  // padding
    
    // Fire module parameters
    int SC; // squeeze channels
    int EC; // expand channels
    int fire_id; // which fire module (2-9)
    
    LayerConfig() : layer_type(LAYER_NONE), H(0), W(0), IC(0), OC(0),
                    K(0), S(0), P(0), SC(0), EC(0), fire_id(0) {}
};

// ============================================================================
// Module Function Declarations
// ============================================================================

void conv3d(
    bool enable,
    fixed_point_t activations[MAX_CONV_H][MAX_CONV_W][MAX_CONV1_IC],
    fixed_point_t weights[MAX_CONV_K][MAX_CONV_K][MAX_CONV1_IC][MAX_CONV1_OC],
    fixed_point_t output[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],
    int H, int W, int IC, int OC, int K, int S, int P
);

void fire(
    bool enable,
    const fixed_point_t input[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],
    const fixed_point_t squeeze_weights[MAX_FIRE_IC][MAX_FIRE_SC],
    const fixed_point_t expand1x1_weights[MAX_FIRE_SC][MAX_FIRE_EC],
    const fixed_point_t expand3x3_weights[3][3][MAX_FIRE_SC][MAX_FIRE_EC],
    fixed_point_t output[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],
    int H, int W, int IC, int SC, int EC
);

void maxpool(
    bool enable,
    const fixed_point_t activations[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],
    fixed_point_t output[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],
    int H, int W, int IC
);

void avgpool(
    bool enable,
    const fixed_point_t activations[AVGPOOL_H][AVGPOOL_W][AVGPOOL_C],
    fixed_point_t output[AVGPOOL_C],
    int H, int W, int IC
);

void conv1(
    bool enable,
    fixed_point_t activations[MAX_CONV_H][MAX_CONV_W][MAX_CONV1_IC],
    fixed_point_t weights[MAX_CONV_K][MAX_CONV_K][MAX_CONV1_IC][MAX_CONV1_OC],
    fixed_point_t output[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],
    int H, int W, int IC, int OC
);

void conv10(
    bool enable,
    fixed_point_t activations[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],
    fixed_point_t weights[MAX_FIRE_IC][AVGPOOL_C],
    fixed_point_t output[AVGPOOL_H][AVGPOOL_W][AVGPOOL_C]
);

// ============================================================================
// Controller Function - Main Entry Point
// ============================================================================

/*
 * Main controller function that executes the entire SqueezeNet pipeline
 * 
 * @param input_image: Input image (224x224x3)
 * @param final_output: Output class scores (10 classes)
 */
void controller(
    fixed_point_t input_image[MAX_CONV_H][MAX_CONV_W][MAX_CONV1_IC],
    fixed_point_t final_output[AVGPOOL_C]
);

/*
 * Configure layer parameters based on layer index
 * 
 * @param layer: Layer index (0-13)
 * @param config: Output configuration structure
 */
void configure_layer(int layer, LayerConfig &config);

/*
 * Execute a configured layer
 * 
 * @param config: Layer configuration
 * @param input_buf: Input activation buffer
 * @param output_buf: Output activation buffer
 * @param avgpool_output: Special output buffer for final avgpool
 */
void execute_layer(
    LayerConfig &config,
    fixed_point_t input_buf[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],
    fixed_point_t output_buf[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],
    fixed_point_t avgpool_output[AVGPOOL_C]
);

#endif // KERNEL_H