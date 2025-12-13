#include "config.h"
#include "kernel.h"

/*
look at squeezenet.ipynb and shapes.txt
imo just declare buf1 and buf2 in this file and reuse them
so first buf1 is input and buf2 is output then switch since each module feeds to the next
need to resize to smaller for avgpool but other than that every module should have input/output buffers same size
*/

void squeezenet(
    float input[MAX_CONV_H * MAX_CONV_W * 3], // 224x224x3 images
    int output[NUM_CLASSES]
)
{
    // Declare two buffers for ping-pong operation
    static fixed_point_t buf1[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC];
    static fixed_point_t buf2[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC];
    
    // Special buffer for avgpool (different dimensions)
    static fixed_point_t avgpool_buf[AVGPOOL_C];
    
    // Weight buffers (these should be loaded from external memory in real implementation)
    static fixed_point_t conv1_weights[MAX_CONV_K][MAX_CONV_K][MAX_CONV_IC][MAX_CONV_OC];
    static fixed_point_t conv10_weights[1][1][MAX_CONV_IC][NUM_CLASSES];
    
    // Fire module weight buffers
    static fixed_point_t fire_squeeze_weights[MAX_FIRE_IC][MAX_FIRE_SC];
    static fixed_point_t fire_expand1x1_weights[MAX_FIRE_SC][MAX_FIRE_EC];
    static fixed_point_t fire_expand3x3_weights[3][3][MAX_FIRE_SC][MAX_FIRE_EC];
    
    // Convert flattened input to 3D format in buf1
    for (int h = 0; h < MAX_CONV_H; h++) {
        for (int w = 0; w < MAX_CONV_W; w++) {
            for (int c = 0; c < 3; c++) {
                buf1[h][w][c] = (fixed_point_t)input[h * MAX_CONV_W * 3 + w * 3 + c];
            }
        }
    }
    
    // Args struct
    Args args;
    
    // Ping-pong between buf1 and buf2
    bool use_buf1_as_input = true;
    
    // Process all layers
    for (int layer = 0; layer < TOTAL_LAYERS; layer++) {
        // Get layer configuration from controller
        controller(layer, args);
        
        // Set input/output buffers based on ping-pong
        if (layer == 13) { // Special case for avgpool
            // AvgPool reads from current buffer but needs special dimensions
            args.avgpool_input = (fixed_point_t (*)[AVGPOOL_W][AVGPOOL_C])(use_buf1_as_input ? buf1 : buf2);
            args.avgpool_output = avgpool_buf;
        } else {
            if (use_buf1_as_input) {
                args.input_buf = buf1;
                args.output_buf = buf2;
            } else {
                args.input_buf = buf2;
                args.output_buf = buf1;
            }
        }
        
        // Load weights for current layer (in real implementation, read from external memory)
        if (layer == 0) {
            args.conv_weights = conv1_weights;
        } else if (layer == 12) {
            args.conv_weights = (fixed_point_t (*)[MAX_CONV_K][MAX_CONV_IC][MAX_CONV_OC])conv10_weights;
        } else if (args.enable_fire) {
            args.squeeze_weights = fire_squeeze_weights;
            args.expand1x1_weights = fire_expand1x1_weights;
            args.expand3x3_weights = fire_expand3x3_weights;
        }
        
        // Execute the appropriate module
        if (args.enable_conv) {
            conv3d(
                true,
                (fixed_point_t (*)[MAX_CONV_W][MAX_CONV_IC])args.input_buf,
                args.conv_weights,
                args.output_buf,
                args.H, args.W, args.IC, args.OC, args.K, args.S, args.P
            );
        }
        
        if (args.enable_maxpool) {
            maxpool(
                true,
                args.input_buf,
                args.output_buf,
                args.H, args.W, args.IC
            );
        }
        
        if (args.enable_fire) {
            fire(
                true,
                args.input_buf,
                args.squeeze_weights,
                args.expand1x1_weights,
                args.expand3x3_weights,
                args.output_buf,
                args.H, args.W, args.IC, args.SC, args.EC
            );
        }
        
        if (args.enable_avgpool) {
            avgpool(
                true,
                args.avgpool_input,
                args.avgpool_output,
                args.H, args.W, args.IC
            );
        }
        
        // Toggle buffer for next layer (except for last layer)
        if (layer < TOTAL_LAYERS - 1) {
            use_buf1_as_input = !use_buf1_as_input;
        }
    }
    
    // Convert avgpool output to integer output
    for (int i = 0; i < NUM_CLASSES; i++) {
        output[i] = (int)avgpool_buf[i];
    }
}