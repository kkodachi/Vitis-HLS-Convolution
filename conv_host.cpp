#include <iostream>
#include <cstdlib>
#include "config.h"
#include "conv3d_kernel.h"

// CIFAR10 Params
#define IC 3       // input channels
#define OC 8       // output channels
#define D 1        // input depth
#define H 32        // input height
#define W 32        // input width
#define Kd 1       // kernel depth
#define Kh 3       // kernel height
#define Kw 3       // kernel width

#define stride_d 1
#define stride_h 1
#define stride_w 1

#define pad_d 0
#define pad_h 0
#define pad_w 0


// derived output dimensions
#define D_out ((D + 2*pad_d - Kd)/stride_d + 1)
#define H_out ((H + 2*pad_h - Kh)/stride_h + 1)
#define W_out ((W + 2*pad_w - Kw)/stride_w + 1)

// flattening helpers
inline int idx4(int d, int h, int w, int c, int D_, int H_, int W_, int C_) {
    return ((c * D_ + d) * H_ + h) * W_ + w;
}

// compare output helper
void compare_output(data_type *output, data_type *golden){
    int errors = 0;
    for(int i = 0; i < OC * D_out * H_out * W_out; i++) {
        if(output[i] != golden[i]) {
            errors++;
            if(errors < 5) {
                std::cout << "Mismatch at index " << i << ": golden=" << (float)golden[i] << " kernel=" << (float)output[i] << std::endl;
            }
        }
        else if (i < 5){
        	std::cout << "Match at " << i << ": " << (float)output[i] << std::endl;
        }
    }

    if(errors == 0) std::cout << "Kernel matches golden" << std::endl;
    else std::cout << errors << " errors found" << std::endl;
}

// golden 3D convolution
void conv_golden(
    data_type *A_ptr, data_type *W_ptr, data_type *Y_ptr,
    int s_d, int s_h, int s_w,
    int p_d, int p_h, int p_w
) {


    for (int oc = 0; oc < OC; oc++) {
        for (int od = 0; od < D_out; od++) {
            for (int oh = 0; oh < H_out; oh++) {
                for (int ow = 0; ow < W_out; ow++) {

                    data_type sum = 0;

                    for (int ic = 0; ic < IC; ic++) {
                        for (int kd = 0; kd < Kd; kd++) {
                            int in_d = od*s_d - p_d + kd;
                            if (in_d < 0 || in_d >= D) continue;

                            for (int kh = 0; kh < Kh; kh++) {
                                int in_h = oh*s_h - p_h + kh;
                                if (in_h < 0 || in_h >= H) continue;

                                for (int kw = 0; kw < Kw; kw++) {
                                    int in_w = ow*s_w - p_w + kw;
                                    if (in_w < 0 || in_w >= W) continue;

                                    size_t a_idx =
                                        ((ic * D + in_d) * H + in_h) * W + in_w;

                                    size_t w_idx =
                                        (((oc * IC + ic) * Kd + kd) * Kh + kh) * Kw + kw;

                                    sum += A_ptr[a_idx] * W_ptr[w_idx];
                                }
                            }
                        }
                    }

                    size_t out_idx =
                        ((oc * D_out + od) * H_out + oh) * W_out + ow;

                    Y_ptr[out_idx] = sum;
                }
            }
        }
    }
}

int main() {
    std::srand(1);

    data_type *activations = new data_type[IC*D*H*W];
    data_type *weights = new data_type[OC*IC*Kd*Kh*Kw];
    data_type *output_ws = new data_type[OC*D_out*H_out*W_out];
    data_type *output_os = new data_type[OC*D_out*H_out*W_out];
    data_type *golden = new data_type[OC*D_out*H_out*W_out];

    // randomize activations and weights
    for(int i = 0; i < IC*D*H*W; i++) {
        int r = std::rand() % 256 - 128;
        activations[i] = data_type(r/16.0);
    }
    for(int i = 0; i < OC*IC*Kd*Kh*Kw; i++) {
        int r = std::rand() % 256 - 128;
        weights[i] = data_type(r/16.0);
    }

    // initialize outputs
    for(int i = 0; i < OC*D_out*H_out*W_out; i++) {
        output_ws[i] = 0;
        output_os[i] = 0;
        golden[i] = 0;
    }

    // compute golden
    conv_golden(
        activations,
        weights,
        golden,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w
    );


    // call kernels
    conv3d_ws(activations, H, W, D, IC, weights, Kh, Kw, Kd, OC, output_ws);
    // conv3d_os(activations, H, W, D, IC, weights, Kh, Kw, Kd, OC, output_os);


    // compare outputs
    std::cout << "Weight stationary kernel:" << std::endl;
    compare_output(output_ws, golden);
    // std::cout << "Output stationary kernel:" << std::endl;
    // compare_output(output_os, golden);

    // Cleanup
    delete[] activations;
    delete[] weights;
    delete[] output_ws;
    delete[] output_os;
    delete[] golden;

    return 0;
}
