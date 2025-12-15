#include "kernel.h"
#include "config.h"

void maxpool(
    bool enable,
    const fixed_point_t activations[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],
    fixed_point_t output[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],
    int H,      // input height
    int W,      // input width
    int IC     // input channels
)
{
    const int K = 3;
    const int S = 2;
    const int P = 0;

    int H_OUT = (H + 2*P - K)/S + 1;
    int W_OUT = (W + 2*P - K)/S + 1;

    fixed_point_t line_buffer[3][MAX_FIRE_W];
    #pragma HLS ARRAY_PARTITION variable=line_buffer dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=line_buffer dim=2 factor=3

    fixed_point_t local_output[MAX_FIRE_W];
    #pragma HLS ARRAY_PARTITION variable=local_output dim=1 factor=3

    IC_LOOP:
    for (int ic = 0; ic < IC;ic++){
        OH_LOOP:
        for (int oh = 0; oh < H_OUT; oh++){
            LOAD_LINES:
            for (int k = 0; k < K; k++) {
                int ih = oh * S + k;
                for (int w = 0; w < W; w++) {
                    #pragma HLS PIPELINE II=1
                    line_buffer[k][w] = (ih < H && w < W) ? activations[ih][w][ic] : (fixed_point_t)0;
                }
            }
            OW_LOOP:
            for (int ow = 0; ow < W_OUT; ow++) {
                #pragma HLS PIPELINE II=1
                fixed_point_t max_val = line_buffer[0][ow * S];

                MAX_LOOP:
                for (int kh = 0; kh < K; kh++) {
                    #pragma HLS UNROLL
                    for (int kw = 0; kw < K; kw++) {
                        #pragma HLS UNROLL
                        int iw = ow * S + kw;
                        if (iw < W) {
                            fixed_point_t val = line_buffer[kh][iw];
                            if (val > max_val) {
                                max_val = val;
                            }
                        }
                    }
                }
                local_output[ow] = max_val;
            }
            WB_LOOP: 
            for (int ow = 0; ow < W_OUT; ow++) {
				#pragma HLS PIPELINE II=1
                output[oh][ow][ic] = local_output[ow];
            }
        }
    }
}

// void maxpool(
//     bool enable,
//     const fixed_point_t activations[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],
//     fixed_point_t output[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],
//     int H,      // input height
//     int W,      // input width
//     int IC     // input channels
// )
// {
//     if (!enable) return;
//     fixed_point_t output_local[MAX_FIRE_H][MAX_FIRE_W];
//     fixed_point_t window[3][3];
//     #pragma HLS ARRAY_PARTITION variable=window complete dim=0

//     const int K = 3;
//     const int S = 2;
//     const int P = 0;

//     int H_OUT = (H + 2*P - K)/S + 1;
//     int W_OUT = (W + 2*P - K)/S + 1;

//     IC_LOOP:
//     for (int ic = 0; ic < IC;ic++){
//         OH_LOOP:
//         for (int oh = 0; oh < H_OUT; oh++){
//             OW_LOOP:
//             for (int ow = 0; ow < W_OUT; ow++){
//                 LOAD_KH:
//                 for (int kh = 0; kh < K; kh++) {
//                     LOAD_KW:
//                     for (int kw = 0; kw < K; kw++) {
//                         #pragma HLS PIPELINE II=1
//                         int ih = oh * S + kh;
//                         int iw = ow * S + kw;
//                         window[kh][kw] = activations[ih][iw][ic];
//                     }
//                 }
                
//                 fixed_point_t max_val = window[0][0];
//                 KH_LOOP:
//                 for (int kh = 0; kh < K; kh++) {
//                     #pragma HLS UNROLL
//                     KW_LOOP:
//                     for (int kw = 0; kw < K; kw++) {
//                         #pragma HLS UNROLL
//                         if (window[kh][kw] > max_val)
//                             max_val = window[kh][kw];
//                     }
//                 }
//                 output_local[oh][ow] = max_val;
//             }
//         }
//         for (int oh = 0; oh < H_OUT; oh++){
//             for (int ow = 0; ow < W_OUT; ow++){
//                 #pragma HLS PIPELINE II=1
//                 output[oh][ow][ic] = output_local[oh][ow];
//             }
//         }
//     }
// }
