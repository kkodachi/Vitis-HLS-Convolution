#include "kernel.h"
#include "config.h"

void squeeze(
    const fixed_point_t input[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],
    const fixed_point_t weights[MAX_FIRE_IC][MAX_FIRE_SC],
    fixed_point_t squeeze_output[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_SC],
    int H, int W, int IC, int SC
)
{
    fixed_point_t input_local[MAX_FIRE_IC];
    #pragma HLS ARRAY_PARTITION variable=input_local cyclic factor=4
    #pragma HLS bind_storage variable=input_local type=ram_2p impl=lutram

    #pragma HLS ARRAY_PARTITION variable=weights cyclic factor=4 dim=1
    #pragma HLS RESOURCE variable=weights core=RAM_2P_BRAM

    H_LOOP:
    for (int h = 0; h < H; h++) {
        W_LOOP:
        for (int w = 0; w < W; w++) {
            LOAD_INPUT:
            for (int ic = 0; ic < IC; ic++) {
                #pragma HLS PIPELINE II=1
                input_local[ic] = input[h][w][ic];
            }

            SC_LOOP:
            for (int sc = 0; sc < SC; sc++) {
                accum_t sum = 0;

                IC_LOOP:
                for (int ic = 0; ic < IC; ic++) {
                    #pragma HLS UNROLL factor=4
                    sum += input_local[ic] * weights[ic][sc];
                }

                squeeze_output[h][w][sc] = (sum > 0) ? (fixed_point_t)sum : (fixed_point_t)0;
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
    fixed_point_t input_local[MAX_FIRE_SC];
    #pragma HLS ARRAY_PARTITION variable=input_local complete
    #pragma HLS bind_storage variable=input_local type=ram_2p impl=lutram

    #pragma HLS ARRAY_PARTITION variable=expand1x1_weights cyclic factor=4 dim=1
    #pragma HLS bind_storage variable=expand1x1_weights type=ram_2p impl=bram

    #pragma HLS bind_storage variable=output type=ram_2p impl=bram

    H_LOOP:
    for (int h = 0; h < H; h++) {
        W_LOOP:
        for (int w = 0; w < W; w++) {
            LOAD_INPUT:
            for (int sc = 0; sc < SC; sc++) {
                #pragma HLS PIPELINE II=1
                input_local[sc] = input[h][w][sc];
            }

            for (int ec = 0; ec < EC; ec++){    
                accum_t sum = 0;

                SC_LOOP:
                for (int sc = 0; sc < SC; sc++) {
                    sum += input_local[sc] * expand1x1_weights[sc][ec];
                }

                output[h][w][ec] = (sum > 0) ? (fixed_point_t)sum : (fixed_point_t)0;
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

    accum_t temp_output[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_EC];
    #pragma HLS ARRAY_PARTITION variable=temp_output cyclic factor=4 dim=3
    
    fixed_point_t input_local[MAX_FIRE_H][MAX_FIRE_W];
    #pragma HLS bind_storage variable=input_local type=ram_2p impl=lutram
    #pragma HLS ARRAY_PARTITION variable=input_local cyclic factor=3 dim=2

    fixed_point_t weights_local[K][K];
    #pragma HLS ARRAY_PARTITION variable=weights_local complete dim=1
    #pragma HLS ARRAY_PARTITION variable=weights_local complete dim=2

    INIT_EC:
    for (int ec = 0; ec < EC; ec++) {
        INIT_H:
        for (int h = 0; h < H_OUT; h++) {
            INIT_W:
            for (int w = 0; w < W_OUT; w++) {
                #pragma HLS PIPELINE II=1
                temp_output[h][w][ec] = 0;
            }
        }
    }

    SC_LOOP:
    for (int sc = 0; sc < SC; sc++){
        LOAD_INPUT:
        for (int hi = 0; hi < H; hi++){
            for (int wi = 0; wi < W; wi++){
                #pragma HLS PIPELINE II=1
                input_local[hi][wi] = input[hi][wi][sc];
            }
        }

        EC_LOOP:
        for (int ec = 0; ec < EC; ec++){
            LOAD_WEIGHTS:
            for (int i = 0; i < K; i++){
                for (int j = 0; j < K; j++){
                    #pragma HLS UNROLL
                    weights_local[i][j] = expand3x3_weights[i][j][sc][ec];
                }
            }

            H_OUT_LOOP:
            for (int h = 0; h < H_OUT; h++) {
                W_OUT_LOOP:
                for (int w = 0; w < W_OUT; w++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS DEPENDENCE variable=temp_output inter false
                    
                    accum_t sum = 0;

                    KH_LOOP:
                    for (int kh = 0; kh < K; kh++) {
                        #pragma HLS UNROLL
                        KW_LOOP:
                        for (int kw = 0; kw < K; kw++) {
                            #pragma HLS UNROLL
                            int h_in = h * S + kh - P;
                            int w_in = w * S + kw - P;

                            fixed_point_t val = 0;
                            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                val = input_local[h_in][w_in];
                            }

                            sum += val * weights_local[kh][kw];
                        }
                    }
                    
                    temp_output[h][w][ec] += sum;
                }
            }
        }
    }

    WB_EC:
    for (int ec = 0; ec < EC; ec++) {
        WB_H:
        for (int h = 0; h < H_OUT; h++) {
            WB_W:
            for (int w = 0; w < W_OUT; w++) {
                #pragma HLS PIPELINE II=1
                accum_t val = temp_output[h][w][ec];
                output[h][w][offset + ec] = (val > 0) ? (fixed_point_t)val : (fixed_point_t)0;
            }
        }
    }
}

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