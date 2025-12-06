#include "kernel.h"

void maxpool(
    const act_t activations[MAX_H * MAX_W * MAX_IC],
    act_t output[MAX_H * MAX_W * MAX_IC],
    int H,      // input height
    int W,      // input width
    int IC,     // input channels
    int K,      // kernel size
    int stride  // stride
)
{
    #pragma HLS INTERFACE m_axi port=activations offset=slave bundle=gmem0 depth=524288
    #pragma HLS INTERFACE m_axi port=output      offset=slave bundle=gmem1 depth=524288

    #pragma HLS INTERFACE s_axilite port=activations
    #pragma HLS INTERFACE s_axilite port=output
    #pragma HLS INTERFACE s_axilite port=H
    #pragma HLS INTERFACE s_axilite port=W
    #pragma HLS INTERFACE s_axilite port=IC
    #pragma HLS INTERFACE s_axilite port=K
    #pragma HLS INTERFACE s_axilite port=stride
    #pragma HLS INTERFACE s_axilite port=return

    // clamp for safety
    if (H > MAX_H) H = MAX_H;
    if (W > MAX_W) W = MAX_W;
    if (IC > MAX_IC) IC = MAX_IC;

    const int outH = (H - K) / stride + 1;
    const int outW = (W - K) / stride + 1;

    H_LOOP:
    for (int oh = 0; oh < outH; oh++){
        W_LOOP:
        for (int ow = 0; ow < outW; ow++){
            // top left of window in the input
            const int h0 = oh * stride;
            const int w0 = ow * stride;

            CH_LOOP:
            for (int c = 0; c < IC; c++){
                #pragma HLS PIPELINE II=1
                // index of first element in window
                int idx0 = h0 * (MAX_W * MAX_IC) + w0 * (MAX_IC) + c;
                fixed_point_t cur_max = activations[idx0];

                KH_LOOP:
                for (int kh = 0; kh < K; kh++) {
                    #pragma HLS UNROLL
                    KW_LOOP:
                    for (int kw = 0; kw < K; kw++) {
                        #pragma HLS UNROLL
                        int ih = h0 + kh;
                        int iw = w0 + kw;
                        if (ih < H && iw < W) {
                            int idx = ih * (MAX_W * MAX_IC) + iw * (MAX_IC) + c;
                            fixed_point_t val = activations[idx];
                            if (val > cur_max) cur_max = val;
                        }
                    }
                }

                // write pooled value to output
                int out_idx = oh*(MAX_W * MAX_IC) + ow*(MAX_IC) + c;
                output[out_idx] = cur_max;
            }
        }
    }
}