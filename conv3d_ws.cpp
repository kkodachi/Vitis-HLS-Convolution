#include "conv3d_kernel.h"

// load weights for output channel into BRAM
#pragma HLS INLINE
void load_weights_oc(
    fixed_point_t weights[MAX_K][MAX_K][MAX_IC][MAX_OC],
    fixed_point_t local_w[MAX_K][MAX_K][MAX_IC],
    int K, int IC, int oc
){
    LOAD_KH: for(int kh = 0; kh < MAX_K; kh++){
        LOAD_KW: for(int kw = 0; kw < MAX_K; kw++){
            LOAD_IC: for(int ic = 0; ic < MAX_IC; ic++){
                #pragma HLS PIPELINE II=1
                if(kh < K && kw < K && ic < IC){
                    local_w[kh][kw][ic] = weights[kh][kw][ic][oc];
                }
            }
        }
    }
}


// weight stationary convolution kernel
void conv2d_ws(
    fixed_point_t activations[MAX_H][MAX_W][MAX_IC],
    fixed_point_t weights[MAX_K][MAX_K][MAX_IC][MAX_OC],
    fixed_point_t output[MAX_H][MAX_W][MAX_OC],

    int H,      // input height
    int W,      // input width
    int IC,     // input channels
    int OC,     // output channels
    int K,      // kernel size
    int stride, // stride
    int pad     // padding
){
    // AXI interfaces
    // #pragma HLS INTERFACE m_axi     port=activations offset=slave depth=32768
    // #pragma HLS INTERFACE m_axi     port=weights     offset=slave depth=65536
    // #pragma HLS INTERFACE m_axi     port=output      offset=slave depth=32768
    #pragma HLS INTERFACE m_axi port=activations offset=slave depth=MAX_H*MAX_W*MAX_IC
    #pragma HLS INTERFACE m_axi port=weights     offset=slave depth=MAX_K*MAX_K*MAX_IC*MAX_OC
    #pragma HLS INTERFACE m_axi port=output      offset=slave depth=MAX_H*MAX_W*MAX_OC


    #pragma HLS INTERFACE s_axilite port=H
    #pragma HLS INTERFACE s_axilite port=W
    #pragma HLS INTERFACE s_axilite port=IC
    #pragma HLS INTERFACE s_axilite port=OC
    #pragma HLS INTERFACE s_axilite port=K
    #pragma HLS INTERFACE s_axilite port=stride
    #pragma HLS INTERFACE s_axilite port=pad
    #pragma HLS INTERFACE s_axilite port=return

    // output dimensions
    const int H_out = (H + 2*pad - K) / stride + 1;
    const int W_out = (W + 2*pad - K) / stride + 1;

    // local weight buffer
    fixed_point_t local_w[MAX_K][MAX_K][MAX_IC];
    #pragma HLS ARRAY_PARTITION variable=local_w dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=local_w dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=local_w dim=3 complete

    
    OC_LOOP:
    for(int oc = 0; oc < OC; oc++){
        
        // load weights into BRAM
        load_weights_oc(weights, local_w, K, IC, oc);

        H_OUT:
        for(int ho = 0; ho < H_out; ho++){
            W_OUT:
            for(int wo = 0; wo < W_out; wo++){
                #pragma HLS PIPELINE II=1

                accum_t sum = 0;

                // input start
                int h_base = ho * stride - pad;
                int w_base = wo * stride - pad;

                //---------------------------------------------------
                // Convolution window (UNROLLED)
                //---------------------------------------------------
                KH:
                for(int kh = 0; kh < MAX_K; kh++){
                    #pragma HLS UNROLL

                    KW:
                    for(int kw = 0; kw < MAX_K; kw++){
                        #pragma HLS UNROLL

                        IC_LOOP:
                        for(int ic = 0; ic < MAX_IC; ic++){
                            #pragma HLS UNROLL

                            if(kh < K && kw < K && ic < IC){
                                int h_in = h_base + kh;
                                int w_in = w_base + kw;

                                // padding check
                                if(h_in >= 0 && h_in < H &&
                                   w_in >= 0 && w_in < W){
                                    sum += activations[h_in][w_in][ic] *
                                           local_w[kh][kw][ic];
                                }
                            }
                        }
                    }
                }

                output[ho][wo][oc] = (fixed_point_t)sum;
            }
        }
    }
}
