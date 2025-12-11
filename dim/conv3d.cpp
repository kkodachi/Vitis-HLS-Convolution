#include "kernel.h"
#include "config.h"

void load_weights(
    fixed_point_t weights[MAX_CONV_K][MAX_CONV_K][MAX_CONV_IC][MAX_CONV_OC],
    fixed_point_t local_weights[MAX_CONV_K][MAX_CONV_K],
    int K, int IC_ind, int OC_ind
){
    LOAD_WEIGHTS:
    for (int kx = 0; kx < K; kx++) {
        for (int ky = 0; ky < K; ky++) {
            #pragma HLS PIPELINE II=1
            local_weights[kx][ky] = weights[kx][ky][IC_ind][OC_ind];
        }
    }
}

void load_activations(
    fixed_point_t activations[MAX_CONV_H][MAX_CONV_W][MAX_CONV_IC],
    fixed_point_t local_activations[MAX_CONV_H + 2*MAX_CONV_K][MAX_CONV_W + 2*MAX_CONV_K],
    int H, int W, int IC_ind, int pad
){
    LOAD_ACTIVATION:
    for (int h = 0; h < H + 2*pad; h++) {
        for (int w = 0; w < W + 2*pad; w++) {
            #pragma HLS PIPELINE II=1
            if (h < pad || h >= H + pad || w < pad || w >= W + pad) {
                local_activations[h][w] = 0; // zero padding
            } else {
                local_activations[h][w] = activations[h - pad][w - pad][IC_ind];
            }
        }
    }
}

void conv3d(
    bool enable,
    fixed_point_t activations[MAX_CONV_H][MAX_CONV_W][MAX_CONV_IC],
    fixed_point_t weights[MAX_CONV_K][MAX_CONV_K][MAX_CONV_IC][MAX_CONV_OC],
    fixed_point_t output[MAX_CONV_H][MAX_CONV_W][MAX_CONV_OC],

    int H,      // input height
    int W,      // input width
    int IC,     // input channels
    int OC,     // output channels
    int K,      // kernel size
    int S,      // stride
    int P       // padding
)
{
    #pragma HLS INTERFACE mode=s_axilite port=H
    #pragma HLS INTERFACE mode=s_axilite port=W
    #pragma HLS INTERFACE mode=s_axilite port=IC
    #pragma HLS INTERFACE mode=s_axilite port=OC
    #pragma HLS INTERFACE mode=s_axilite port=K
    #pragma HLS INTERFACE mode=s_axilite port=stride
    #pragma HLS INTERFACE mode=s_axilite port=pad
    #pragma HLS INTERFACE mode=s_axilite port=return

    fixed_point_t local_weights[MAX_CONV_K][MAX_CONV_K];

    fixed_point_t local_activations[MAX_CONV_H + 2*MAX_CONV_K][MAX_CONV_W + 2*MAX_CONV_K];

    fixed_point_t local_output[MAX_CONV_H][MAX_CONV_W];

    int H_OUT = (H + 2*pad - K)/stride + 1;
    int W_OUT = (W + 2*pad - K)/stride + 1;

    for (int oc=0;oc<OC;oc++){

        INIT_LOCAL_OUTPUT:
        for (int h = 0; h < H_OUT; h++) {
            for (int w = 0; w < W_OUT; w++) {
                #pragma HLS PIPELINE II=1
                local_output[h][w] = 0;
            }
        }

        IC_LOOP:
        for (int ic=0;ic<IC;ic++){
            load_weights(weights,local_weights,K,ic,oc);
            load_activations(activations,local_activations,H,W,ic,P);

            H_OUT_LOOP:
            for (int h = 0; h < H_OUT; h++) {
                W_OUT_LOOP:
                for (int w = 0; w < W_OUT; w++) {
                    #pragma HLS PIPELINE II=1

                    accum_t sum = 0;

                    KH_LOOP:
                    for (int kh = 0; kh < K; kh++) {
                        #pragma HLS UNROLL
                        KW_LOOP:
                        for (int kw = 0; kw < K; kw++) {
                            #pragma HLS UNROLL
                            int h_in = h * S + kh;
                            int w_in = w * S + kw;

                            sum += local_activations[h_in][w_in] *
                                   local_weights[kh][kw];
                        }
                    }

                    local_output[h][w] += (fixed_point_t)sum;
                }
            }
        }

        WB_LOOP:
        for (int h = 0; h < H_OUT; h++) {
            for (int w = 0; w < W_OUT; w++) {
                #pragma HLS PIPELINE II=1
                output[h][w][oc] = local_output[h][w];
            }
        }
    }
}