#include "kernel.h"

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
)
{
    if (!enable) return;

    #pragma HLS INLINE off

    #pragma HLS INTERFACE s_axilite port=enable
    #pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0 depth=262144
    #pragma HLS INTERFACE m_axi port=squeeze_weights offset=slave bundle=gmem1 depth=262144
    #pragma HLS INTERFACE m_axi port=expand1x1_weights offset=slave bundle=gmem2 depth=262144
    #pragma HLS INTERFACE m_axi port=expand3x3_weights offset=slave bundle=gmem3 depth=589824
    #pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem4 depth=262144
    #pragma HLS INTERFACE s_axilite port=H
    #pragma HLS INTERFACE s_axilite port=W
    #pragma HLS INTERFACE s_axilite port=IC
    #pragma HLS INTERFACE s_axilite port=squeeze_ch
    #pragma HLS INTERFACE s_axilite port=expand_ch
    #pragma HLS INTERFACE s_axilite port=return

    fixed_point_t squeeze_out[MAX_H * MAX_W * MAX_IC];
    #pragma HLS ARRAY_PARTITION variable=squeeze_out cyclic factor=8 dim=1
    
    fixed_point_t expand1x1_out[MAX_H * MAX_W * MAX_OC];
    #pragma HLS ARRAY_PARTITION variable=expand1x1_out cyclic factor=8 dim=1
    
    fixed_point_t expand3x3_out[MAX_H * MAX_W * MAX_OC];
    #pragma HLS ARRAY_PARTITION variable=expand3x3_out cyclic factor=8 dim=1

    // Initialize arrays - use separate non-overlapping loops
    INIT_SQUEEZE:
    for (int i = 0; i < MAX_H * MAX_W * MAX_IC; i++) {
        #pragma HLS PIPELINE II=1
        squeeze_out[i] = 0;
    }
    
    INIT_EXPAND1X1:
    for (int i = 0; i < MAX_H * MAX_W * MAX_OC; i++) {
        #pragma HLS PIPELINE II=1
        expand1x1_out[i] = 0;
    }
    
    INIT_EXPAND3X3:
    for (int i = 0; i < MAX_H * MAX_W * MAX_OC; i++) {
        #pragma HLS PIPELINE II=1
        expand3x3_out[i] = 0;
    }

    // squeeze: 1x1 conv
    conv3d(true, (fixed_point_t*)input, (fixed_point_t*)squeeze_weights, 
           squeeze_out, H, W, IC, squeeze_ch, 1, 1, 0);
    
    // relu after squeeze
    relu(true, squeeze_out, H, W, squeeze_ch);

    // expand 1x1: 1x1 conv
    conv3d(true, squeeze_out, (fixed_point_t*)expand1x1_weights, 
           expand1x1_out, H, W, squeeze_ch, expand_ch, 1, 1, 0);

    // expand 3x3: 3x3 conv with pad=1
    conv3d(true, squeeze_out, (fixed_point_t*)expand3x3_weights, 
           expand3x3_out, H, W, squeeze_ch, expand_ch, 3, 1, 1);

    // concatenate expand1x1 and expand3x3 along channel dimension
    // output channels = expand_ch + expand_ch = 2*expand_ch
    CONCAT_H:
    for (int h = 0; h < H; h++){
        #pragma HLS LOOP_TRIPCOUNT min=4 max=32
        
        CONCAT_W:
        for (int w = 0; w < W; w++){
            #pragma HLS LOOP_TRIPCOUNT min=4 max=32
            
            // first half: expand1x1
            CONCAT_C1:
            for (int c = 0; c < expand_ch; c++){
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=64 max=256
                
                int idx_in = h * (MAX_W * MAX_OC) + w * (MAX_OC) + c;
                int idx_out = h * (MAX_W * MAX_OC) + w * (MAX_OC) + c;
                output[idx_out] = expand1x1_out[idx_in];
            }
            // second half: expand3x3
            CONCAT_C2:
            for (int c = 0; c < expand_ch; c++){
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=64 max=256
                
                int idx_in = h * (MAX_W * MAX_OC) + w * (MAX_OC) + c;
                int idx_out = h * (MAX_W * MAX_OC) + w * (MAX_OC) + expand_ch + c;
                output[idx_out] = expand3x3_out[idx_in];
            }
        }
    }

    // relu on concatenated output
    relu(true, output, H, W, expand_ch * 2);
}



