#include "kernel.h"

void load_weights(
    const fixed_point_t weights[MAX_K * MAX_K * MAX_IC * MAX_OC],
    fixed_point_t local_weights[MAX_K * MAX_K * MAX_IC],
    int IC, int K, int oc
){
    LOAD_WEIGHTS:
    for (int kx = 0; kx < K; kx++) {
        for (int ky = 0; ky < K; ky++) {
            for (int ic = 0; ic < IC; ic++) {

                #pragma HLS PIPELINE II=1

                int idx = kx * (MAX_K * MAX_IC * MAX_OC)
                        + ky * (MAX_IC * MAX_OC)
                        + ic * (MAX_OC)
                        + oc;

                int local_idx = kx * (MAX_K * MAX_IC)
                              + ky * (MAX_IC)
                              + ic;

                local_weights[local_idx] = weights[idx];
            }
        }
    }
}

void load_activations(
    const fixed_point_t activations[MAX_H * MAX_W * MAX_IC],
    fixed_point_t local_activations[(MAX_H + 2*MAX_K)*(MAX_W + 2*MAX_K)],
    int H, int W, int ic, int pad
){
    LOAD_ACTIVATION:
    for (int h = 0; h < H + 2*pad; h++) {
        for (int w = 0; w < W + 2*pad; w++) {

            #pragma HLS PIPELINE II=1

            int local_idx = h * (MAX_W + 2*MAX_K) + w;

            if (h < pad || h >= H + pad || w < pad || w >= W + pad) {
                local_activations[local_idx] = 0; // zero padding
            } else {
                int h_in = h - pad;
                int w_in = w - pad;
                int idx = h_in * (MAX_W * MAX_IC) + w_in * (MAX_IC) + ic;
                local_activations[local_idx] = activations[idx];
            }
        }
    }
}

/*
activations[H][W][IC], idx = h * (MAX_W * MAX_IC) + w * MAX_IC + c
weights[K][K][IC][OC]
output[HOUT][W_OUT[OC]]
*/
void conv3d(
    bool enable,
    fixed_point_t activations[MAX_H * MAX_W * MAX_IC],
    fixed_point_t weights[MAX_K * MAX_K * MAX_IC *MAX_OC],
    fixed_point_t output[MAX_H * MAX_W * MAX_OC],

    int H,      // input height
    int W,      // input width
    int IC,     // input channels
    int OC,     // output channels
    int K,      // kernel size
    int stride, // stride
    int pad     // padding
)
{
    if (!enable) return;

    #pragma HLS INLINE off

    #pragma HLS INTERFACE s_axilite port=enable
    #pragma HLS INTERFACE s_axilite port=H
    #pragma HLS INTERFACE s_axilite port=W
    #pragma HLS INTERFACE s_axilite port=IC
    #pragma HLS INTERFACE s_axilite port=OC
    #pragma HLS INTERFACE s_axilite port=K
    #pragma HLS INTERFACE s_axilite port=stride
    #pragma HLS INTERFACE s_axilite port=pad
    #pragma HLS INTERFACE s_axilite port=return

    #pragma HLS INTERFACE m_axi port=activations offset=slave depth=262144
    #pragma HLS INTERFACE m_axi port=weights     offset=slave depth=589824
    #pragma HLS INTERFACE m_axi port=output      offset=slave depth=262144

    // Local buffers with array partitioning for parallelism
    fixed_point_t local_weights[MAX_K * MAX_K * MAX_IC];
    #pragma HLS ARRAY_PARTITION variable=local_weights cyclic factor=8 dim=1
    
    fixed_point_t local_activations[(MAX_H + 2*MAX_K)*(MAX_W + 2*MAX_K)];
    #pragma HLS ARRAY_PARTITION variable=local_activations cyclic factor=8 dim=1
    
    fixed_point_t local_output[MAX_H * MAX_W];
    #pragma HLS ARRAY_PARTITION variable=local_output cyclic factor=8 dim=1

    int H_OUT = (H + 2*pad - K)/stride + 1;
    int W_OUT = (W + 2*pad - K)/stride + 1;

    OC_LOOP:
    for (int oc=0;oc<OC;oc++){
        #pragma HLS LOOP_TRIPCOUNT min=10 max=512
        
        load_weights(weights,local_weights,IC,K,oc);

        INIT_ZERO_LOOP:
        for (int h = 0; h < H_OUT; h++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=32
            for (int w = 0; w < W_OUT; w++) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=32
                #pragma HLS PIPELINE II=1
                int out_idx = h * MAX_W + w;
                local_output[out_idx] = 0;
            }
        }

        IC_LOOP:
        for (int ic=0;ic<IC;ic++){
            #pragma HLS LOOP_TRIPCOUNT min=3 max=512
            
            load_activations(activations,local_activations,H,W,ic,pad);
            
            H_OUT_LOOP:
            for (int h=0;h<H_OUT;h++){
                #pragma HLS LOOP_TRIPCOUNT min=1 max=32
                
                W_OUT_LOOP:
                for (int w=0;w<W_OUT;w++){
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=32
                    
                    accum_t sum = 0;

                    KH_LOOP:
                    for (int kh = 0; kh < K; kh++) {
                        #pragma HLS UNROLL
                        KW_LOOP:
                        for (int kw = 0; kw < K; kw++) {
                            #pragma HLS UNROLL
                            #pragma HLS PIPELINE II=1
                            
                            int h_in = h * stride + kh;
                            int w_in = w * stride + kw;

                            int local_idx = h_in * (MAX_W + 2*MAX_K) + w_in;
                            int w_idx = kh * (MAX_K * MAX_IC) + kw * (MAX_IC) + ic;

                            sum += local_activations[local_idx] * local_weights[w_idx];
                        }
                    }
                    int out_idx = h * (MAX_W) + w;
                    local_output[out_idx] += (fixed_point_t)sum;
                }
            }
        }

        WB_LOOP:
        for (int h = 0; h < H_OUT; h++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=32
            for (int w = 0; w < W_OUT; w++) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=32
                #pragma HLS PIPELINE II=1
                int out_idx = h * (MAX_W * MAX_OC) + w * (MAX_OC) + oc;
                int local_idx = h * MAX_W + w;
                output[out_idx] = local_output[local_idx];
            }
        }
    }
}