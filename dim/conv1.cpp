#include "config.h"
#include "kernel.h"

#define MAX_FIRE_H 112
#define MAX_FIRE_W 112
#define MAX_FIRE_IC 512 // max input channels
#define MAX_FIRE_SC 64 // max squeeze channels
#define MAX_FIRE_EC 256 // max expand channels

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
//    const int H = 224;
//    const int W = 224;
//    const int IC = 3;
//    const int OC = 96;
    const int K = 7;
    const int S = 2;
    const int P = 3;
    int H_OUT = (H + 2*P - K)/S + 1;
    int W_OUT = (W + 2*P - K)/S + 1;

    if (!enable) return;

    // removed static from buffers
    
    static fixed_point_t line_buffer[MAX_CONV_K][MAX_CONV_W + 2*MAX_CONV_K];
    #pragma HLS ARRAY_PARTITION variable=line_buffer dim=1 complete
//    #pragma HLS ARRAY_PARTITION variable=line_buffer dim=2 complete
	#pragma HLS ARRAY_PARTITION variable=line_buffer dim=2 factor=7

    fixed_point_t kernel[MAX_CONV_K][MAX_CONV_K];
    #pragma HLS ARRAY_PARTITION variable=kernel dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=kernel dim=2 complete

    accum_t psum[MAX_FIRE_W];
//    #pragma HLS ARRAY_PARTITION variable=psum dim=1 complete
	#pragma HLS ARRAY_PARTITION variable=psum dim=1 factor=7

    OC_LOOP:
	for (int oc = 0; oc < OC; oc++) {
        IC_LOOP:
		for (int ic = 0; ic < IC; ic++) {
            LOAD_WEIGHTS:
            for (int kh = 0; kh < K; kh++){
                for (int kw = 0; kw < K; kw++){
                	#pragma HLS PIPELINE II=1
                    kernel[kh][kw] = weights[kh][kw][ic][oc];
                }
            }
            OH_LOOP:
            for (int oh = 0; oh < H_OUT; oh++) {
                LOAD_LINE:
                for (int k = 0; k < K; k++) {
                    int ih = oh * S + k - P;
                    for (int w = 0; w < W + 2*P; w++) {
                    	#pragma HLS PIPELINE II=1
                        int iw = w - P;
                        line_buffer[k][w] = (ih >= 0 && ih < H && iw >= 0 && iw < W) ? 
                                            activations[ih][iw][ic] : (fixed_point_t)0;
                    }
                }

                INIT_PSUM:
                for (int ow = 0; ow < W_OUT; ow++) {
                    #pragma HLS PIPELINE II=1
                    psum[ow] = 0;
                }
                
                OW_LOOP:
                for (int ow = 0; ow < W_OUT; ow++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS DEPENDENCE variable=output inter false
                    accum_t acc = 0;
                    MAC_LOOP:
                    for (int kh = 0; kh < K; kh++) {
                        #pragma HLS UNROLL
                        for (int kw = 0; kw < K; kw++) {
                            #pragma HLS UNROLL
                            acc += line_buffer[kh][ow * S + kw] * kernel[kh][kw];
                        }
                    }

                    psum[ow] = acc;
                }
                WB_LOOP:
                for (int ow = 0; ow < W_OUT; ow++) {
                    #pragma HLS PIPELINE II=1
                    output[oh][ow][oc] = psum[ow];
                }
            }
        }
    }
}

//void conv1(
//    bool enable,
//    fixed_point_t activations[MAX_CONV_H][MAX_CONV_W][MAX_CONV1_IC],
//    fixed_point_t weights[MAX_CONV_K][MAX_CONV_K][MAX_CONV1_IC][MAX_CONV1_OC],
//    fixed_point_t output[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],
//    int H,
//    int W,
//    int IC,
//    int OC
//)
//{
//    const int K = 7;
//    const int S = 2;
//    const int P = 3;
//    int H_OUT = (H + 2*P - K)/S + 1;
//    int W_OUT = (W + 2*P - K)/S + 1;
//
//    #pragma HLS INTERFACE mode=s_axilite port=H
//    #pragma HLS INTERFACE mode=s_axilite port=W
//    #pragma HLS INTERFACE mode=s_axilite port=IC
//    #pragma HLS INTERFACE mode=s_axilite port=OC
//    #pragma HLS INTERFACE mode=s_axilite port=return
//
//    // fully partitioned weight buffer for one OC
//    fixed_point_t local_weights[K][K][MAX_CONV1_IC];
//    #pragma HLS ARRAY_PARTITION variable=local_weights complete dim=1
//    #pragma HLS ARRAY_PARTITION variable=local_weights complete dim=2
//    #pragma HLS ARRAY_PARTITION variable=local_weights complete dim=3
//
//    // local activation buffer for one output pixel
//    fixed_point_t local_activations[K][K][MAX_CONV1_IC];
//    #pragma HLS ARRAY_PARTITION variable=local_activations complete dim=1
//    #pragma HLS ARRAY_PARTITION variable=local_activations complete dim=2
//    #pragma HLS ARRAY_PARTITION variable=local_activations complete dim=3
//
//    accum_t accum_output[MAX_FIRE_H][MAX_FIRE_W];
//    #pragma HLS ARRAY_PARTITION variable=accum_output complete dim=2
//
//    OC_LOOP:
//    for (int oc = 0; oc < OC; oc++) {
//    	CLEAR_ACCUM:
//    	for (int h = 0; h < H_OUT; h++) {
//    		for (int w = 0; w < W_OUT; w++) {
//    			#pragma HLS PIPELINE
//    	        accum_output[h][w] = 0;
//    		}
//    	}
//        IC_LOOP:
//        for (int ic = 0; ic < IC; ic++) {
//            // Load weights for this output channel
//            LOAD_WEIGHTS:
//            for (int kh = 0; kh < K; kh++) {
//                for (int kw = 0; kw < K; kw++) {
//                	#pragma HLS PIPELINE II=1
//                    local_weights[kh][kw][ic] = weights[kh][kw][ic][oc];
//                }
//            }
//
//
//
//            // Convolve per output pixel
//            H_OUT_LOOP:
//            for (int h = 0; h < H_OUT; h++) {
//                W_OUT_LOOP:
//                for (int w = 0; w < W_OUT; w++) {
//                	accum_t current_accum = accum_output[h][w];
//
//                    // Load KxKxIC patch into local_activations
//                    LOAD_LOCAL:
//                    for (int kh = 0; kh < K; kh++) {
//                        for (int kw = 0; kw < K; kw++) {
//                            #pragma HLS PIPELINE II=1
//                            int h_in = h*S + kh - P;
//                            int w_in = w*S + kw - P;
//
//                            fixed_point_t val = 0;
//                            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W)
//                                val = activations[h_in][w_in][ic];
//
//                            local_activations[kh][kw][ic] = val;
//                        }
//                    }
//                    // fully unroll MAC loop
//                    accum_t sum = 0;
//                    MAC_LOOP:
//                    for (int kh = 0; kh < K; kh++) {
//                        #pragma HLS UNROLL
//                        for (int kw = 0; kw < K; kw++) {
//                            #pragma HLS UNROLL
//                            sum += local_activations[kh][kw][ic] * local_weights[kh][kw][ic];
//                        }
//                    }
//
//                    current_accum += sum;
//                    accum_output[h][w] = current_accum;
//                }
//            }
//        }
//
//        WB_LOOP:
//        for (int h = 0; h < H_OUT; h++) {
//            for (int w = 0; w < W_OUT; w++) {
//                #pragma HLS PIPELINE II=1
//                output[h][w][oc] = (fixed_point_t)accum_output[h][w];
//            }
//        }
//    }
//}
