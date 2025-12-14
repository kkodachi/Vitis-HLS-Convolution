#include "config.h"
#include "kernel.h"

void conv1(
    bool enable,
    fixed_point_t activations[MAX_CONV_H][MAX_CONV_W][MAX_CONV1_IC],
    fixed_point_t weights[MAX_CONV_K][MAX_CONV_K][MAX_CONV1_IC][MAX_CONV1_OC],
    fixed_point_t output[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],
    int H,
    int W,
    int IC,
    int OC
)
{
    const int K = 7;
    const int S = 2;
    const int P = 3;
    int H_OUT = (H + 2*P - K)/S + 1;
    int W_OUT = (W + 2*P - K)/S + 1;

    #pragma HLS INTERFACE mode=s_axilite port=H
    #pragma HLS INTERFACE mode=s_axilite port=W
    #pragma HLS INTERFACE mode=s_axilite port=IC
    #pragma HLS INTERFACE mode=s_axilite port=OC
    #pragma HLS INTERFACE mode=s_axilite port=return

    // fully partitioned weight buffer for one OC
    fixed_point_t local_weights[K][K][MAX_CONV1_IC];
    #pragma HLS ARRAY_PARTITION variable=local_weights complete dim=1
    #pragma HLS ARRAY_PARTITION variable=local_weights complete dim=2
    #pragma HLS ARRAY_PARTITION variable=local_weights complete dim=3

    // local activation buffer for one output pixel
    fixed_point_t local_activations[K][K][MAX_CONV1_IC];
    #pragma HLS ARRAY_PARTITION variable=local_activations complete dim=1
    #pragma HLS ARRAY_PARTITION variable=local_activations complete dim=2
    #pragma HLS ARRAY_PARTITION variable=local_activations complete dim=3

    accum_t accum_output[MAX_FIRE_H][MAX_FIRE_W];
    #pragma HLS ARRAY_PARTITION variable=accum_output complete dim=2

    OC_LOOP:
    for (int oc = 0; oc < OC; oc++) {
    	CLEAR_ACCUM:
    	for (int h = 0; h < H_OUT; h++) {
    		for (int w = 0; w < W_OUT; w++) {
    			#pragma HLS PIPELINE
    	        accum_output[h][w] = 0;
    		}
    	}
        IC_LOOP:
        for (int ic = 0; ic < IC; ic++) {
            // Load weights for this output channel
            LOAD_WEIGHTS:
            for (int kh = 0; kh < K; kh++) {
                for (int kw = 0; kw < K; kw++) {
                	#pragma HLS PIPELINE II=1
                    local_weights[kh][kw][ic] = weights[kh][kw][ic][oc];
                }
            }



            // Convolve per output pixel
            H_OUT_LOOP:
            for (int h = 0; h < H_OUT; h++) {
                W_OUT_LOOP:
                for (int w = 0; w < W_OUT; w++) {
                	accum_t current_accum = accum_output[h][w];

                    // Load KxKxIC patch into local_activations
                    LOAD_LOCAL:
                    for (int kh = 0; kh < K; kh++) {
                        for (int kw = 0; kw < K; kw++) {
                            #pragma HLS PIPELINE II=1
                            int h_in = h*S + kh - P;
                            int w_in = w*S + kw - P;

                            fixed_point_t val = 0;
                            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W)
                                val = activations[h_in][w_in][ic];

                            local_activations[kh][kw][ic] = val;
                        }
                    }
                    // fully unroll MAC loop
                    accum_t sum = 0;
                    MAC_LOOP:
                    for (int kh = 0; kh < K; kh++) {
                        #pragma HLS UNROLL
                        for (int kw = 0; kw < K; kw++) {
                            #pragma HLS UNROLL
                            sum += local_activations[kh][kw][ic] * local_weights[kh][kw][ic];
                        }
                    }

                    current_accum += sum;
                    accum_output[h][w] = current_accum;
                }
            }
        }

        WB_LOOP:
        for (int h = 0; h < H_OUT; h++) {
            for (int w = 0; w < W_OUT; w++) {
                #pragma HLS PIPELINE II=1
                output[h][w][oc] = (fixed_point_t)accum_output[h][w];
            }
        }
    }
}
