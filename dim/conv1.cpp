#include "config.h"
#include "kernel.h"

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
)
{
    const int K = 7;    // kernel size
    const int S = 2;    // stride
    const int P = 3;    // padding
    int H_OUT = (H + 2*P - K)/S + 1; // 56
    int W_OUT = (W + 2*P - K)/S + 1; // 56
    
    #pragma HLS INTERFACE mode=s_axilite port=H
    #pragma HLS INTERFACE mode=s_axilite port=W
    #pragma HLS INTERFACE mode=s_axilite port=IC
    #pragma HLS INTERFACE mode=s_axilite port=OC
    #pragma HLS INTERFACE mode=s_axilite port=S
    #pragma HLS INTERFACE mode=s_axilite port=P
    #pragma HLS INTERFACE mode=s_axilite port=return

    fixed_point_t local_weights[MAX_CONV_K][MAX_CONV_K][MAX_CONV1_IC];
    #pragma HLS ARRAY_PARTITION variable=local_weights cyclic factor=16 dim=3
	#pragma HLS ARRAY_PARTITION variable=local_weights complete dim=1
	#pragma HLS ARRAY_PARTITION variable=local_weights complete dim=2

    fixed_point_t local_activations[MAX_CONV_H + 2*MAX_CONV_K][MAX_CONV_W + 2*MAX_CONV_K];
    #pragma HLS ARRAY_PARTITION variable=local_activations cyclic factor=3 dim=2

    fixed_point_t local_output[MAX_FIRE_H][MAX_FIRE_W];
    #pragma HLS ARRAY_PARTITION variable=local_output complete dim=2

    accum_t accum_output[MAX_FIRE_H][MAX_FIRE_W];
    #pragma HLS ARRAY_PARTITION variable=accum_output complete dim=2

    OC_LOOP:
    for (int oc=0;oc<OC;oc++){
        INIT_ZERO_LOOP:
        for (int h = 0; h < H_OUT; h++) {
            for (int w = 0; w < W_OUT; w++) {
                #pragma HLS PIPELINE II=1
                accum_output[h][w] = 0;
            }
        }

        IC_LOOP:
        for (int ic=0;ic<IC;ic++){
            LOAD_WEIGHTS:
            for (int kh = 0; kh < K; kh++){
                #pragma HLS UNROLL
                for (int kw = 0; kw < K; kw++){
                    #pragma HLS UNROLL
                    local_weights[kh][kw][ic] = weights[kh][kw][ic][oc];
                }
            }

            LOAD_ACTIVATIONS:
            for (int h = 0; h < H + 2*P; h++){
                for (int w = 0; w < W + 2*P; w++){
                    #pragma HLS PIPELINE II=1
                    if (h < P || h >= H + P || w < P || w >= W + P)
                        local_activations[h][w] = 0;
                    else
                        local_activations[h][w] = activations[h - P][w - P][ic];
                }
            }

            H_OUT_LOOP:
            for (int h=0;h<H_OUT;h++){
                W_OUT_LOOP:
                for (int w=0;w<W_OUT;w++){
                    #pragma HLS PIPELINE II=1
                    #pragma HLS DEPENDENCE variable=accum_output inter false
                    
                    accum_t partial = 0;

                    KH_LOOP:
                    for (int kh = 0; kh < K; kh++) {
                        #pragma HLS UNROLL
                        KW_LOOP:
                        for (int kw = 0; kw < K; kw++) {
                            #pragma HLS UNROLL
                            int h_in = h * S + kh;
                            int w_in = w * S + kw;

                            partial += local_activations[h_in][w_in] * 
                                       local_weights[kh][kw][ic];
                        }
                    }

                    accum_output[h][w] += partial;
                }
            }
        }

        WRITE_OUTPUT:
        for (int h = 0; h < H_OUT; h++){
            for (int w = 0; w < W_OUT; w++){
                #pragma HLS PIPELINE II=1
                local_output[h][w] = accum_output[h][w];
            }
        }

        WB_LOOP:
        for (int h = 0; h < H_OUT; h++){
            for (int w = 0; w < W_OUT; w++){
                #pragma HLS PIPELINE II=1
                output[h][w][oc] = local_output[h][w];
            }
        }
    }
}