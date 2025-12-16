#include "config.h"
#include "kernel.h"

void conv(
    fixed_point_t activations[MAX_CONV_DIM][MAX_CONV_DIM][MAX_CONV_DIM],
    fixed_point_t weights[MAX_CONV_K][MAX_CONV_K][MAX_CONV_DIM][MAX_CONV_DIM],
    fixed_point_t output[MAX_CONV_DIM][MAX_CONV_DIM][MAX_CONV_DIM],
     int H,
     int W,
     int IC,
     int OC
)
{
    const int K = 3;
    const int S = 2;
    const int P = 1;
    int H_OUT = (H + 2*P - K)/S + 1;
    int W_OUT = (W + 2*P - K)/S + 1;

    // removed static from buffers
    
    static fixed_point_t line_buffer[MAX_CONV_DIM][MAX_CONV_DIM + 2*MAX_CONV_K];
    #pragma HLS ARRAY_PARTITION variable=line_buffer dim=1 complete
//    #pragma HLS ARRAY_PARTITION variable=line_buffer dim=2 complete
	#pragma HLS ARRAY_PARTITION variable=line_buffer dim=2 factor=3

    fixed_point_t kernel[MAX_CONV_K][MAX_CONV_K];
    #pragma HLS ARRAY_PARTITION variable=kernel dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=kernel dim=2 complete

    accum_t psum[MAX_CONV_DIM];
//    #pragma HLS ARRAY_PARTITION variable=psum dim=1 complete
	#pragma HLS ARRAY_PARTITION variable=psum dim=1 factor=3

    OC_LOOP:
	for (int oc = 0; oc < OC; oc++) {
        OH_LOOP:
        for (int oh = 0; oh < H_OUT; oh++) {
            INIT_PSUM:
            for (int ow = 0; ow < W_OUT; ow++) {
                #pragma HLS PIPELINE II=1
                psum[ow] = 0;
            }
            IC_LOOP:
		    for (int ic = 0; ic < IC; ic++) {
                LOAD_WEIGHTS:
                for (int kh = 0; kh < K; kh++){
                    for (int kw = 0; kw < K; kw++){
                	    #pragma HLS PIPELINE II=1
                        kernel[kh][kw] = weights[kh][kw][ic][oc];
                    }
                }
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

                    psum[ow] += acc;
                }
            }
            WB_LOOP:
            for (int ow = 0; ow < W_OUT; ow++) {
                #pragma HLS PIPELINE II=1
                // output[oh][ow][oc] = ((fixed_point_t)psum[ow] > 0) ? (fixed_point_t)psum[ow] : 0;
                accum_t val = psum[ow];
                val = val < 0 ? 0 : (val > 127 ? 127 : val);
                output[oh][ow][oc] = (fixed_point_t)val;
            }
        }
    }
}
