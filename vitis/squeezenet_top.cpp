#include "kernel.h"

void squeezenet_top(
    fixed_point_t input[MAX_H * MAX_W * MAX_IC],
    fixed_point_t output[10],
    fixed_point_t* all_weights,
    int num_stages
)
{
    #pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0 depth=262144
    #pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1 depth=10
    #pragma HLS INTERFACE m_axi port=all_weights offset=slave bundle=gmem2 depth=10000000
    #pragma HLS INTERFACE s_axilite port=num_stages
    #pragma HLS INTERFACE s_axilite port=return

    // intermediate buffers
    static fixed_point_t buffer1[MAX_H * MAX_W * MAX_OC];
    static fixed_point_t buffer2[MAX_H * MAX_W * MAX_OC];

    // enable signals
    bool en_conv, en_maxpool, en_avgpool, en_relu, en_fire;

    // current working buffers
    fixed_point_t* current_input = input;
    fixed_point_t* current_output = buffer1;

    // weight pointers (simplified - in practice, slice all_weights appropriately)
    fixed_point_t* conv_weights = all_weights;
    fixed_point_t* fire_squeeze_weights = all_weights;
    fixed_point_t* fire_expand1x1_weights = all_weights;
    fixed_point_t* fire_expand3x3_weights = all_weights;

    // layer dimensions (example for CIFAR-10 upsampled to 32x32)
    int H = 32;
    int W = 32;
    int IC = 3;
    int OC = 64;

    // main loop - iterate through stages
    STAGE_LOOP:
    for (int stage = 0; stage < num_stages; stage++) {
        #pragma HLS LOOP_TRIPCOUNT min=16 max=16
        
        // get enable signals from controller
        controller(stage, &en_conv, &en_maxpool, &en_avgpool, &en_relu, &en_fire);

        // execute enabled module
        conv3d(en_conv, current_input, conv_weights, current_output, 
               H, W, IC, OC, 3, 1, 1);
        
        maxpool(en_maxpool, current_input, current_output, 
                H, W, IC, 2, 2);
        
        avgpool(en_avgpool, current_input, current_output, 
                H, W, IC, 13, 1);
        
        relu(en_relu, current_input, H, W, IC);
        
        fire_module(en_fire, current_input, fire_squeeze_weights,
                    fire_expand1x1_weights, fire_expand3x3_weights,
                    current_output, H, W, IC, 16, 64);

        // swap buffers
        if (current_input == input) {
            current_input = buffer1;
            current_output = buffer2;
        } else if (current_input == buffer1) {
            current_input = buffer2;
            current_output = buffer1;
        } else {
            current_input = buffer1;
            current_output = buffer2;
        }
    }

    // final classification (copy to output)
    for (int i = 0; i < 10; i++) {
        #pragma HLS PIPELINE II=1
        output[i] = current_input[i];
    }
}
