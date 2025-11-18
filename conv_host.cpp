#include "config.h"
#include "conv3d_kernel.h"

const int OUT_H = H - K + 1;
const int OUT_W = W - K + 1;

void compare_output(data_type *output, data_type *golden){
    int errors = 0;
    for(int i = 0; i < OUT_H*OUT_W*OC; i++) {
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

void conv_golden(ap_fixed<8,4> *A_ptr, ap_fixed<8,4> *W_ptr, ap_fixed<8,4> *Y_ptr) {
    for(int oh = 0; oh < OUT_H; oh++) {
        for(int ow = 0; ow < OUT_W; ow++) {
            for(int oc = 0; oc < OC; oc++) {
                ap_fixed<24,12> sum = 0;
                for(int kh = 0; kh < K; kh++) {
                    for(int kw = 0; kw < K; kw++) {
                        for(int ic = 0; ic < IC; ic++) {
                            sum += A_ptr[idx3(oh+kh, ow+kw, ic, W, IC)] *
                                   W_ptr[idx4(kh, kw, ic, oc, K, IC, OC)];
                        }
                    }
                }
                Y_ptr[idx3(oh, ow, oc, OUT_W, OC)] = ap_fixed<8,4>(sum);
            }
        }
    }
}

// COPIED FROM HW3, NEED TO CHANGE FOR 3D CONV
int main() {
    std::srand(1);

    data_type *activations = new data_type[H*W*IC];
    data_type *weights = new data_type[K*K*IC*OC];
    data_type *output_ws = new data_type[OUT_H*OUT_W*OC];
    data_type *output_os = new data_type[OUT_H*OUT_W*OC];
    data_type *golden1 = new data_type[OUT_H*OUT_W*OC];

    // randomize activations and weights for Q4.4 range -128 to 127
    for(int i = 0; i < H*W*IC; i++) {
        int r = std::rand() % 256 - 128; 
        activations[i] = data_type(r/16.0);
    }
    for(int i = 0; i < K*K*IC*OC; i++) {
        int r = std::rand() % 256 - 128;
        weights[i] = data_type(r/16.0);
    }

    // init output
    for(int i = 0; i < OUT_H*OUT_W*OC; i++){
    	output_ws[i] = 0;
        output_os[i] = 0;
    	golden[i] = 0;
    }

    // compute golden
    conv_golden(activations, weights, golden);

    // call kernel
    
    // compare outputs
    std::cout << "Weight stationary kernel:" << std::endl;
    compare_output(output_ws,golden);
    std::cout << "Output stationary kernel:" << std::endl;
    compare_output(output_os,golden);

    // deallocate
    delete[] activations;
    delete[] weights;
    delete[] output_ws;
    delete[] output_os;
    delete[] golden;

    return 0;
}