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

    // intermediate buffers - remove static to avoid stack overflow in testbench
    fixed_point_t buffer1[MAX_H * MAX_W * MAX_OC];
    fixed_point_t buffer2[MAX_H * MAX_W * MAX_OC];

    // enable signals
    bool en_conv, en_maxpool, en_avgpool, en_relu, en_fire;

    // current working buffers
    fixed_point_t* current_input = input;
    fixed_point_t* current_output = buffer1;

    // layer dimensions from controller
    int H_in, W_in, IC_in;
    int H_out, W_out, OC_out;
    
    // layer-specific parameters from controller
    int K, stride, pad;
    int squeeze_ch, expand_ch;
    int weight_offset;

    // main loop - iterate through stages
    STAGE_LOOP:
    for (int stage = 0; stage < num_stages; stage++) {
        #pragma HLS LOOP_TRIPCOUNT min=16 max=16
        
        // get ALL configuration from controller
        controller(stage, &en_conv, &en_maxpool, &en_avgpool, &en_relu, &en_fire,
                   &H_in, &W_in, &IC_in, &H_out, &W_out, &OC_out,
                   &K, &stride, &pad, &squeeze_ch, &expand_ch, &weight_offset);

        // weight pointers with proper offsets
        fixed_point_t* stage_weights = all_weights + weight_offset;
        
        // for fire modules, calculate offsets for three weight sets
        int squeeze_weight_size = 1 * 1 * IC_in * squeeze_ch;
        int expand1x1_weight_size = 1 * 1 * squeeze_ch * expand_ch;
        
        fixed_point_t* fire_squeeze_weights = stage_weights;
        fixed_point_t* fire_expand1x1_weights = stage_weights + squeeze_weight_size;
        fixed_point_t* fire_expand3x3_weights = stage_weights + squeeze_weight_size + expand1x1_weight_size;

        // execute enabled module - use dimensions directly from controller
        if (en_conv) {
            conv3d(true, current_input, stage_weights, current_output, 
                   H_in, W_in, IC_in, OC_out, K, stride, pad);
            // swap buffers
            fixed_point_t* temp = current_input;
            current_input = current_output;
            current_output = temp;
        }
        
        if (en_maxpool) {
            maxpool(true, current_input, current_output, 
                    H_in, W_in, IC_in, K, stride);
            // swap buffers
            fixed_point_t* temp = current_input;
            current_input = current_output;
            current_output = temp;
        }
        
        if (en_avgpool) {
            avgpool(true, current_input, current_output, 
                    H_in, W_in, IC_in, K, stride);
            // swap buffers
            fixed_point_t* temp = current_input;
            current_input = current_output;
            current_output = temp;
        }
        
        if (en_relu) {
            // relu is in-place, no buffer swap needed
            relu(true, current_input, H_in, W_in, IC_in);
        }
        
        if (en_fire) {
            fire_module(true, current_input, fire_squeeze_weights,
                        fire_expand1x1_weights, fire_expand3x3_weights,
                        current_output, H_in, W_in, IC_in, squeeze_ch, expand_ch);
            // swap buffers
            fixed_point_t* temp = current_input;
            current_input = current_output;
            current_output = temp;
        }
    }

    // final classification (copy to output)
    // after all stages, should be 1x1x10
    for (int i = 0; i < 10; i++) {
        #pragma HLS PIPELINE II=1
        output[i] = current_input[i];
    }
}



