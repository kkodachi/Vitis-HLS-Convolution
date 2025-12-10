#include "config.h"
#include "kernel.h"

void avgpool(
    const fixed_point_t activations[MAX_H * MAX_W * MAX_IC],
    fixed_point_t output[MAX_H * MAX_W * MAX_IC],
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

    fixed_point_t window[MAX_K * MAX_K * MAX_IC];
    // #pragma HLS ARRAY_PARTITION variable=window complete dim=1

    H_LOOP:
    for (int oh = 0; oh < outH; oh++){
        W_LOOP:
        for (int ow = 0; ow < outW; ow++){
            const int h0 = oh * stride;
            const int w0 = ow * stride;

            CH_LOAD:
            for (int kh = 0; kh < K; kh++) {
                for (int kw = 0; kw < K; kw++) {
                    for (int c = 0; c < IC; c++) {
                        #pragma HLS PIPELINE II=1
                        int ih = h0 + kh;
                        int iw = w0 + kw;

                        int local_idx = c * K * K + kh * K + kw;
                        int idx = ih * (MAX_W * MAX_IC) + iw * MAX_IC + c;

                        if (ih < H && iw < W)
                            window[local_idx] = activations[idx];
                        else
                            window[local_idx] = 0; // padding
                    }
                }
            }

            CH_LOOP:
            for (int c = 0; c < IC; c++) {
                accum_t sum = 0;

                SUM_LOOP:
                for (int i = 0; i < K * K; i++) {
                    #pragma HLS PIPELINE II=1
                    int local_idx = c * K * K + i;
                    sum += window[local_idx];
                }

                // compute average
                fixed_point_t avg = sum / (K * K);

                // write pooled value to output
                int out_idx = oh * (MAX_W * MAX_IC) + ow * MAX_IC + c;
                output[out_idx] = avg;
            }
        }
    }
}



