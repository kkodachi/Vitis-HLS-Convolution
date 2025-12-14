#include "kernel.h"
#include "config.h"

#include <assert.h>

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

    // LOAD_ACTIVATION:
    // for (int h = 0; h < MAX_CONV_H + 2*MAX_CONV_K; h++) {
    //     for (int w = 0; w < MAX_CONV_W + 2*MAX_CONV_K; w++) {
    //         #pragma HLS PIPELINE II=1
    //         if (h < pad || h >= H + pad || w < pad || w >= W + pad) {
    //             local_activations[h][w] = 0; // zero padding
    //         } else {
    //             local_activations[h][w] = activations[h - pad][w - pad][IC_ind];
    //         }
    //     }
    // }
}

void load_activations2(
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

void conv3d_2(
    bool enable,
    fixed_point_t activations[MAX_CONV_H][MAX_CONV_W][MAX_CONV_IC],
    fixed_point_t weights[MAX_CONV_K][MAX_CONV_K][MAX_CONV_IC][MAX_CONV_OC],
    // fixed_point_t output[MAX_CONV_H][MAX_CONV_W][MAX_CONV_OC],
    fixed_point_t output[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],

    int H,      // input height
    int W,      // input width
    int IC,     // input channels
    int OC,     // output channels
    int K,      // kernel size
    int S,      // stride
    int P       // padding
)
{
    if (!enable) return;
    #pragma HLS INTERFACE mode=s_axilite port=H
    #pragma HLS INTERFACE mode=s_axilite port=W
    #pragma HLS INTERFACE mode=s_axilite port=IC
    #pragma HLS INTERFACE mode=s_axilite port=OC
    #pragma HLS INTERFACE mode=s_axilite port=K
    #pragma HLS INTERFACE mode=s_axilite port=S
    #pragma HLS INTERFACE mode=s_axilite port=P
    #pragma HLS INTERFACE mode=s_axilite port=return

    fixed_point_t local_weights[MAX_CONV_K][MAX_CONV_K];
    #pragma HLS ARRAY_PARTITION variable=local_weights complete dim=1
    #pragma HLS ARRAY_PARTITION variable=local_weights complete dim=2

    fixed_point_t local_activations[MAX_CONV_H + 2*MAX_CONV_K][MAX_CONV_W + 2*MAX_CONV_K];
    #pragma HLS ARRAY_PARTITION variable=local_activations cyclic factor=7 dim=2
    #pragma HLS bind_storage variable=local_activations type=ram_2p impl=bram

    // fixed_point_t local_output[MAX_CONV_H][MAX_CONV_W];
    accum_t local_output[MAX_FIRE_H][MAX_FIRE_W];

    // auto local_weights = new fixed_point_t[MAX_CONV_K][MAX_CONV_K];
    // auto local_activations = new fixed_point_t[MAX_CONV_H + 2*MAX_CONV_K][MAX_CONV_W + 2*MAX_CONV_K];
    // auto local_output = new fixed_point_t[MAX_FIRE_H][MAX_FIRE_W];

    int H_OUT = (H + 2*P - K)/S + 1;
    int W_OUT = (W + 2*P - K)/S + 1;

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
                        #pragma HLS UNROLL factor=7
                        #pragma HLS LOOP_TRIPCOUNT min=1 max=7
                        KW_LOOP:
                        for (int kw = 0; kw < K; kw++) {
                            #pragma HLS UNROLL factor=7
                            #pragma HLS LOOP_TRIPCOUNT min=1 max=7
                            int h_in = h * S + kh;
                            int w_in = w * S + kw;

                            sum += local_activations[h_in][w_in] *
                                   local_weights[kh][kw];
                        }
                    }

                    local_output[h][w] += sum;
                }
            }
        }

        WB_LOOP:
        for (int h = 0; h < H_OUT; h++) {
            for (int w = 0; w < W_OUT; w++) {
                #pragma HLS PIPELINE II=1
                output[h][w][oc] = (fixed_point_t)local_output[h][w];
            }
        }
    }
    // delete[] local_weights;
    // delete[] local_activations;
    // delete[] local_output;
}

#include "kernel.h"
#include "config.h"
#include <assert.h>

#define TILE_H 32
#define TILE_W 32

void conv3d(
    bool enable,
    fixed_point_t activations[MAX_CONV_H][MAX_CONV_W][MAX_CONV_IC],
    fixed_point_t weights[MAX_CONV_K][MAX_CONV_K][MAX_CONV_IC][MAX_CONV_OC],
    fixed_point_t output[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],
    int H, int W, int IC, int OC, int K, int S, int P
) {
    if (!enable) return;

    #pragma HLS INTERFACE mode=s_axilite port=H
    #pragma HLS INTERFACE mode=s_axilite port=W
    #pragma HLS INTERFACE mode=s_axilite port=IC
    #pragma HLS INTERFACE mode=s_axilite port=OC
    #pragma HLS INTERFACE mode=s_axilite port=K
    #pragma HLS INTERFACE mode=s_axilite port=S
    #pragma HLS INTERFACE mode=s_axilite port=P
    #pragma HLS INTERFACE mode=s_axilite port=return

    fixed_point_t local_weights[MAX_CONV_K][MAX_CONV_K];
    #pragma HLS ARRAY_PARTITION variable=local_weights complete dim=1
    #pragma HLS ARRAY_PARTITION variable=local_weights complete dim=2

    accum_t local_output[TILE_H][TILE_W];
    #pragma HLS ARRAY_PARTITION variable=local_output complete dim=2
    #pragma HLS DEPENDENCE variable=local_output inter false

    fixed_point_t local_activations[TILE_H + 2*MAX_CONV_K][TILE_W + 2*MAX_CONV_K];
    #pragma HLS ARRAY_PARTITION variable=local_activations cyclic factor=K dim=2
    #pragma HLS bind_storage variable=local_activations type=ram_2p impl=lutram

    int H_OUT = (H + 2*P - K)/S + 1;
    int W_OUT = (W + 2*P - K)/S + 1;

    for (int oc = 0; oc < OC; oc++) {
        for (int tile_h = 0; tile_h < H_OUT; tile_h += TILE_H) {
            for (int tile_w = 0; tile_w < W_OUT; tile_w += TILE_W) {

                // Initialize local_output tile
                for (int h = 0; h < TILE_H && (tile_h + h) < H_OUT; h++) {
                    for (int w = 0; w < TILE_W && (tile_w + w) < W_OUT; w++) {
                        #pragma HLS PIPELINE II=1
                        local_output[h][w] = 0;
                    }
                }

                for (int ic = 0; ic < IC; ic++) {
                    // Load weights for this IC-OC pair
                    for (int kh = 0; kh < K; kh++) {
                        for (int kw = 0; kw < K; kw++) {
                            #pragma HLS UNROLL
                            local_weights[kh][kw] = weights[kh][kw][ic][oc];
                        }
                    }

                    // Load input tile with padding
                    for (int h = 0; h < TILE_H + 2*P && (tile_h + h) < H + 2*P; h++) {
                        for (int w = 0; w < TILE_W + 2*P && (tile_w + w) < W + 2*P; w++) {
                            #pragma HLS PIPELINE II=1
                            int h_idx = tile_h + h - P;
                            int w_idx = tile_w + w - P;
                            if (h_idx < 0 || h_idx >= H || w_idx < 0 || w_idx >= W)
                                local_activations[h][w] = 0;
                            else
                                local_activations[h][w] = activations[h_idx][w_idx][ic];
                        }
                    }

                    // Convolution
                    for (int h = 0; h < TILE_H && (tile_h + h) < H_OUT; h++) {
                        for (int w = 0; w < TILE_W && (tile_w + w) < W_OUT; w++) {
                            #pragma HLS PIPELINE II=1
                            accum_t sum = 0;
                            for (int kh = 0; kh < K; kh++) {
                                #pragma HLS UNROLL
                                for (int kw = 0; kw < K; kw++) {
                                    #pragma HLS UNROLL
                                    int h_in = h * S + kh;
                                    int w_in = w * S + kw;
                                    sum += local_activations[h_in][w_in] * local_weights[kh][kw];
                                }
                            }
                            local_output[h][w] += sum;
                        }
                    }
                }

                // Write back to output
                for (int h = 0; h < TILE_H && (tile_h + h) < H_OUT; h++) {
                    for (int w = 0; w < TILE_W && (tile_w + w) < W_OUT; w++) {
                        #pragma HLS PIPELINE II=1
                        output[tile_h + h][tile_w + w][oc] =
                            (fixed_point_t)((local_output[h][w] > 0) ? local_output[h][w] : 0);
                    }
                }

            } // tile_w
        } // tile_h
    } // oc
}
