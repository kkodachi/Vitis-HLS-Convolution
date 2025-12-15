#include "kernel.h"
#include "config.h"

void squeeze(
    const fixed_point_t input[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],
    const fixed_point_t weights[MAX_FIRE_IC][MAX_FIRE_SC],
    fixed_point_t squeeze_output[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_SC],
    int H, int W, int IC, int SC
)
{
    SC_LOOP:
    for (int sc = 0; sc < SC; sc++) {
        H_LOOP:
        for (int h = 0; h < H; h++) {
            W_LOOP:
            for (int w = 0; w < W; w++) {
                accum_t psum = 0;
                IC_LOOP:
                for (int ic = 0; ic < IC; ic++) {
                    #pragma HLS PIPELINE II=1
                    psum += input[h][w][ic] * weights[ic][sc];
                }
                squeeze_output[h][w][sc] = (fixed_point_t)psum;
            }
        }
    }
}

void expand1(
    const fixed_point_t input[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_SC],
    const fixed_point_t expand1x1_weights[MAX_FIRE_SC][MAX_FIRE_EC],
    fixed_point_t output[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],
    int H, int W, int SC, int EC
)
{
    EC_LOOP:
    for (int ec = 0; ec < EC; ec++) {
        H_LOOP:
        for (int h = 0; h < H; h++) {
            W_LOOP:
            for (int w = 0; w < W; w++) {
                accum_t psum = 0;
                SC_LOOP:
                for (int sc = 0; sc < SC; sc++) {
                    #pragma HLS PIPELINE II=1
                    psum += input[h][w][sc] * expand1x1_weights[sc][ec];
                }
                output[h][w][ec] = (fixed_point_t)psum;
            }
        }
    }
}

void expand3(
    const fixed_point_t input[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_SC],
    const fixed_point_t expand3x3_weights[3][3][MAX_FIRE_SC][MAX_FIRE_EC],
    fixed_point_t output[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],
    int H, int W, int SC, int EC, int offset
)
{
    const int K = 3;
    const int S = 1;
    const int P = 1;

    int H_OUT = (H + 2*P - K)/S + 1;
    int W_OUT = (W + 2*P - K)/S + 1;

    fixed_point_t line_buffer[3][MAX_FIRE_W];
    #pragma HLS ARRAY_PARTITION variable=line_buffer dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=line_buffer dim=2 factor=3

    fixed_point_t kernel[3][3];
    #pragma HLS ARRAY_PARTITION variable=kernel complete dim=1
    #pragma HLS ARRAY_PARTITION variable=kernel complete dim=2

    accum_t psum[MAX_FIRE_W];
	#pragma HLS ARRAY_PARTITION variable=psum dim=1 factor=3

    OC_LOOP:
	for (int oc = 0; oc < EC; oc++) {
        IC_LOOP:
		for (int ic = 0; ic < SC; ic++) {
            LOAD_WEIGHTS:
            for (int kh = 0; kh < K; kh++){
                for (int kw = 0; kw < K; kw++){
                	#pragma HLS PIPELINE II=1
                    kernel[kh][kw] = expand3x3_weights[kh][kw][ic][oc];
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
                                            input[ih][iw][ic] : (fixed_point_t)0;
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
                    output[oh][ow][oc+offset] = psum[ow];
                }
            }
        }
    }
}

// void expand3(
//     const fixed_point_t input[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_SC],
//     const fixed_point_t expand3x3_weights[3][3][MAX_FIRE_SC][MAX_FIRE_EC],
//     fixed_point_t output[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],
//     int H, int W, int SC, int EC, int offset
// )
// {
//     const int K = 3;
//     const int S = 1;
//     const int P = 1;

//     int H_OUT = (H + 2*P - K)/S + 1;
//     int W_OUT = (W + 2*P - K)/S + 1;

//     fixed_point_t input_local[MAX_FIRE_H][MAX_FIRE_W];
//     #pragma HLS bind_storage variable=input_local type=ram_2p impl=lutram
//     #pragma HLS ARRAY_PARTITION variable=input_local cyclic factor=3 dim=2

//     fixed_point_t weights_local[K][K];
//     #pragma HLS ARRAY_PARTITION variable=weights_local complete dim=1
//     #pragma HLS ARRAY_PARTITION variable=weights_local complete dim=2

//     EC_LOOP:
//     for (int ec = 0; ec < EC; ec++){
//         H_OUT_LOOP:
//         for (int h = 0; h < H_OUT; h++) {
//             W_OUT_LOOP:
//             for (int w = 0; w < W_OUT; w++) {
//                 accum_t sum = 0;
//                 SC_LOOP:
//                 for (int sc = 0; sc < SC; sc++){
//                     LOAD_WEIGHTS:
//                     for (int i = 0; i < K; i++){
//                         #pragma HLS UNROLL
//                         for (int j = 0; j < K; j++){
//                             #pragma HLS UNROLL
//                             weights_local[i][j] = expand3x3_weights[i][j][sc][ec];
//                         }
//                     }

//                     LOAD_INPUT:
//                     for (int hi = 0; hi < H; hi++){
//                         for (int wi = 0; wi < W; wi++){
//                             #pragma HLS PIPELINE II=1
//                             input_local[hi][wi] = input[hi][wi][sc];
//                         }
//                     }

//                     KH_LOOP:
//                     for (int kh = 0; kh < K; kh++) {
//                         #pragma HLS UNROLL
//                         KW_LOOP:
//                         for (int kw = 0; kw < K; kw++) {
//                             #pragma HLS UNROLL
//                             int h_in = h * S + kh - P;
//                             int w_in = w * S + kw - P;

//                             fixed_point_t val = 0;
//                             if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
//                                 val = input_local[h_in][w_in];
//                             }

//                             sum += val * weights_local[kh][kw];
//                         }
//                     }
//                 }
//                 output[h][w][offset + ec] = (sum > 0) ? (fixed_point_t)sum : (fixed_point_t)0;
//             }
//         }
//     }
// }

void fire(
    bool enable,
    const fixed_point_t input[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],
    const fixed_point_t squeeze_weights[MAX_FIRE_IC][MAX_FIRE_SC],
    const fixed_point_t expand1x1_weights[MAX_FIRE_SC][MAX_FIRE_EC],
    const fixed_point_t expand3x3_weights[3][3][MAX_FIRE_SC][MAX_FIRE_EC],
    fixed_point_t output[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],
    int H,
    int W,
    int IC,
    int SC,
    int EC
)
{
    if (!enable) return;

    fixed_point_t squeeze_output[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_SC];

    squeeze(input,squeeze_weights,squeeze_output,H,W,IC,SC);
    expand1(squeeze_output,expand1x1_weights,output,H,W,SC,EC);
    expand3(squeeze_output,expand3x3_weights,output,H,W,SC,EC,EC);
}
