#include "config.h"
#include "conv3d_kernel.h"


#define IC 3       // input channels
#define OC 2       // output channels
#define D 5        // input depth
#define H 8        // input height
#define W 8        // input width
#define Kd 3       // kernel depth
#define Kh 3       // kernel height
#define Kw 3       // kernel width

// derived output dimensions
#define D_out ((D + 2*1 - Kd)/1 + 1)
#define H_out ((H + 2*1 - Kh)/1 + 1)
#define W_out ((W + 2*1 - Kw)/1 + 1)

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
                std::cout << "Mismatch at index " << i << ": golden=" << golden[i] << " kernel=" << output[i] << std::endl;
            }
        }
    }

    if(errors == 0) std::cout << "Kernel matches golden" << std::endl;
    else std::cout << errors << " errors found" << std::endl;
}

// golden 3D convolution
void conv_golden(data_type *A_ptr, data_type *W_ptr, data_type *Y_ptr) {
    for(int od = 0; od < D_out; od++) {
        for(int oh = 0; oh < H_out; oh++) {
            for(int ow = 0; ow < W_out; ow++) {
                for(int oc = 0; oc < OC; oc++) {
                    data_type sum = 0;
                    for(int ic = 0; ic < IC; ic++) {
                        for(int kd = 0; kd < Kd; kd++) {
                            int in_d = od + kd;
                            for(int kh = 0; kh < Kh; kh++) {
                                int in_h = oh + kh;
                                for(int kw = 0; kw < Kw; kw++) {
                                    int in_w = ow + kw;
                                    sum += A_ptr[idx4(in_d, in_h, in_w, ic, D, H, W, IC)] *
                                           W_ptr[idx4(kd, kh, kw, oc*IC + ic, Kd, Kh, Kw, OC*IC)];
                                }
                            }
                        }
                    }
                    Y_ptr[idx4(od, oh, ow, oc, D_out, H_out, W_out, OC)] = sum;
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
    conv_golden(activations, weights, golden);

    // call kernels
    conv3d_ws(activations, H, W, D, IC, weights, Kh, Kw, Kd, OC, output_ws);

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
