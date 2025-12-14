#include "config.h"
#include "conv3d_kernel.h"

void conv3d_os(
    fixed_point_t activations[MAX_H][MAX_W][MAX_IC],
    fixed_point_t weights[MAX_K][MAX_K][MAX_IC][MAX_OC],
    fixed_point_t output[MAX_H][MAX_W][MAX_OC],

    int H,      // input height
    int W,      // input width
    int IC,     // input channels
    int OC,     // output channels
    int stride, // stride
    int pad     // padding
)
{
    const int K = 3;
    #pragma HLS INTERFACE mode=s_axilite port=H
    #pragma HLS INTERFACE mode=s_axilite port=W
    #pragma HLS INTERFACE mode=s_axilite port=IC
    #pragma HLS INTERFACE mode=s_axilite port=OC
    // #pragma HLS INTERFACE mode=s_axilite port=K
    #pragma HLS INTERFACE mode=s_axilite port=stride
    #pragma HLS INTERFACE mode=s_axilite port=pad
    #pragma HLS INTERFACE mode=s_axilite port=return

    // MAX_H*MAX_W*MAX_IC
    #pragma HLS INTERFACE m_axi port=activations offset=slave depth=262144
	// MAX_K*MAX_K*MAX_IC*MAX_OC
    #pragma HLS INTERFACE m_axi port=weights     offset=slave depth=589824
	// MAX_H*MAX_W*MAX_OC
    #pragma HLS INTERFACE m_axi port=output      offset=slave depth=262144

    fixed_point_t local_weights[MAX_K][MAX_K][MAX_IC];
    // IC
    #pragma HLS ARRAY_PARTITION variable=local_weights cyclic factor=16 dim=3
	#pragma HLS ARRAY_PARTITION variable=local_weights complete dim=1
	#pragma HLS ARRAY_PARTITION variable=local_weights complete dim=2
    // #pragma HLS ARRAY_PARTITION variable=local_weights cyclic factor=K dim=1
    // #pragma HLS ARRAY_PARTITION variable=local_weights cyclic factor=K dim=2

    fixed_point_t local_activations[MAX_H + 2*MAX_K][MAX_W + 2*MAX_K];
    #pragma HLS ARRAY_PARTITION variable=local_activations cyclic factor=3 dim=2

    fixed_point_t local_output[MAX_H][MAX_W];
    #pragma HLS ARRAY_PARTITION variable=local_output complete dim=2

    // output dimensions
    int H_OUT = (H + 2*pad - K)/stride + 1;
    int W_OUT = (W + 2*pad - K)/stride + 1;

    OC_LOOP:
    for (int oc=0;oc<OC;oc++){
        // init this output tile
        INIT_ZERO_LOOP:
        for (int h = 0; h < H_OUT; h++) {
            for (int w = 0; w < W_OUT; w++) {
                #pragma HLS PIPELINE II=1
                local_output[h][w] = 0;
            }
        }

        IC_LOOP:
        for (int ic=0;ic<IC;ic++){
            // load weights for this IC and OC
            LOAD_WEIGHTS:
            for (int kh = 0; kh < K; kh++){
                for (int kw = 0; kw < K; kw++){
                    #pragma HLS PIPELINE II=1
                    local_weights[kh][kw][ic] = weights[kh][kw][ic][oc];
                }
            }

            // load activations with padding
            LOAD_ACTIVATIONS:
            for (int h = 0; h < H + 2*pad; h++){
                for (int w = 0; w < W + 2*pad; w++){
                    #pragma HLS PIPELINE II=1
                    if (h < pad || h >= H + pad || w < pad || w >= W + pad)
                        local_activations[h][w] = 0;
                    else
                        local_activations[h][w] = activations[h - pad][w - pad][ic];
                }
            }

            H_OUT_LOOP:
            for (int h=0;h<H_OUT;h++){
                W_OUT_LOOP:
                for (int w=0;w<W_OUT;w++){
                    #pragma HLS PIPELINE II=1

                    accum_t sum = local_output[h][w];

                    KH_LOOP:
                    for (int kh = 0; kh < K; kh++) {
                        #pragma HLS UNROLL
                        KW_LOOP:
                        for (int kw = 0; kw < K; kw++) {
                            #pragma HLS UNROLL
                            int h_in = h * stride + kh;
                            int w_in = w * stride + kw;

                            sum += local_activations[h_in][w_in] * 
                                   local_weights[kh][kw][ic];
                        }
                    }

                    local_output[h][w] = sum;
                }
            }
        }

        // write back output
        WB_LOOP:
        for (int h = 0; h < H_OUT; h++){
            for (int w = 0; w < W_OUT; w++){
                #pragma HLS PIPELINE II=1
                output[h][w][oc] = local_output[h][w];
            }
        }
    }
}