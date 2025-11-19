#include <iostream>
#include <cmath>
#include "config.h"
#include "conv3d_kernel.h"
using namespace std;

int main() {
    // REDUCED test dimensions to fit MAX sizes
    const size_t H = 5;
    const size_t W = 5;
    const size_t D = 5;
    const size_t IC = 2;

    const size_t Kh = 3;
    const size_t Kw = 3;
    const size_t Kd = 3;
    const size_t OC = 2;

    const size_t stride_d = 1;
    const size_t stride_h = 1;
    const size_t stride_w = 1;

    const size_t pad_d = 0;
    const size_t pad_h = 0;
    const size_t pad_w = 0;

    const size_t H_out = (H + 2*pad_h - Kh) / stride_h + 1;
    const size_t W_out = (W + 2*pad_w - Kw) / stride_w + 1;
    const size_t D_out = (D + 2*pad_d - Kd) / stride_d + 1;

    const size_t in_size  = H * W * D * IC;
    const size_t w_size   = OC * IC * Kh * Kw * Kd;
    const size_t out_size = H_out * W_out * D_out * OC;

    data_type *activations  = new data_type[in_size];
    data_type *weights      = new data_type[w_size];
    data_type *output_ws    = new data_type[out_size];
    data_type *output_gold  = new data_type[out_size];

    // Initialize activations
    for (size_t c = 0; c < IC; c++) {
        for (size_t d = 0; d < D; d++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    size_t idx =
                        ((c * D + d) * H + h) * W + w;
                    activations[idx] = (data_type)((c + 1) * (d + h + w + 1));
                }
            }
        }
    }

    // Initialize weights
    for (size_t oc = 0; oc < OC; oc++) {
        for (size_t ic = 0; ic < IC; ic++) {
            for (size_t kd = 0; kd < Kd; kd++) {
                for (size_t kh = 0; kh < Kh; kh++) {
                    for (size_t kw = 0; kw < Kw; kw++) {
                        size_t idx =
                            (((oc * IC + ic) * Kd + kd) * Kh + kh) * Kw + kw;
                        weights[idx] = (data_type)(oc + ic + kd + kh + kw + 1);
                    }
                }
            }
        }
    }

    // Golden model
    for (size_t oc = 0; oc < OC; oc++) {
        for (size_t od = 0; od < D_out; od++) {
            for (size_t oh = 0; oh < H_out; oh++) {
                for (size_t ow = 0; ow < W_out; ow++) {
                    data_type sum = 0;
                    for (size_t ic = 0; ic < IC; ic++) {
                        for (size_t kd = 0; kd < Kd; kd++) {
                            long in_d = od * stride_d - pad_d + kd;
                            if (in_d < 0 || in_d >= (long)D) continue;

                            for (size_t kh = 0; kh < Kh; kh++) {
                                long in_h = oh * stride_h - pad_h + kh;
                                if (in_h < 0 || in_h >= (long)H) continue;

                                for (size_t kw = 0; kw < Kw; kw++) {
                                    long in_w = ow * stride_w - pad_w + kw;
                                    if (in_w < 0 || in_w >= (long)W) continue;

                                    size_t a_idx =
                                        ((ic * D + in_d) * H + in_h) * W + in_w;

                                    size_t w_idx =
                                        (((oc * IC + ic) * Kd + kd) * Kh + kh) * Kw + kw;

                                    sum += activations[a_idx] * weights[w_idx];
                                }
                            }
                        }
                    }
                    size_t out_idx =
                        (((oc * D_out) + od) * H_out + oh) * W_out + ow;
                    output_gold[out_idx] = sum;
                }
            }
        }
    }

    // Hardware WS kernel call
    conv3d_ws(
        activations, H, W, D, IC,
        weights, Kh, Kw, Kd, OC,
        output_ws,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w
    );

    // Compare outputs
    int errors = 0;
    for (size_t i = 0; i < out_size; i++) {
        double diff = (double)(output_ws[i] - output_gold[i]);
        if (fabs(diff) > 1e-3) {
            cout << "Mismatch at " << i
                 << ": " << (double)output_ws[i]
                 << " vs " << (double)output_gold[i] << endl;
            errors++;
        }
    }

    if (errors == 0)
        cout << "Kernel matches golden" << endl;
    else
        cout << "Total mismatches: " << errors << endl;

    delete[] activations;
    delete[] weights;
    delete[] output_ws;
    delete[] output_gold;

    return 0;
}



