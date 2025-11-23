#include <iostream>
// #include <cmath>
#include <cstdlib>
#include "conv3d_kernel.h"
// using namespace std;

// golden result for comparison
void conv3d_golden(
    fixed_point_t activations[MAX_H][MAX_W][MAX_IC],
    fixed_point_t weights[MAX_K][MAX_K][MAX_IC][MAX_OC],
    fixed_point_t golden[MAX_H][MAX_W][MAX_OC],
    int H, int W, int IC, int OC, int K,
    int stride,
    int pad
){
    // compute output shape
    int H_out = (H + 2*pad - K) / stride + 1;
    int W_out = (W + 2*pad - K) / stride + 1;

    // initialize output
    for (int oc = 0; oc < OC; oc++) {
        for (int h = 0; h < H_out; h++) {
            for (int w = 0; w < W_out; w++) {
                golden[h][w][oc] = 0;
            }
        }
    }

    // compute convolution
    for (int oc = 0; oc < OC; oc++) {
        for (int oh = 0; oh < H_out; oh++) {
            for (int ow = 0; ow < W_out; ow++) {

                accum_t sum = 0;

                int h_in_origin = oh * stride - pad;
                int w_in_origin = ow * stride - pad;

                for (int kh = 0; kh < K; kh++) {
                    for (int kw = 0; kw < K; kw++) {
                        for (int ic = 0; ic < IC; ic++) {

                            int h_in = h_in_origin + kh;
                            int w_in = w_in_origin + kw;

                            // check if inside padded input
                            if (h_in >= 0 && h_in < H &&
                                w_in >= 0 && w_in < W)
                            {
                                sum += activations[h_in][w_in][ic] *
                                       weights[kh][kw][ic][oc];
                            }
                        }
                    }
                }

                golden[oh][ow][oc] = (fixed_point_t)sum;
            }
        }
    }
}

void compare_results(
		fixed_point_t out_kernel[MAX_H][MAX_W][MAX_OC],
		fixed_point_t golden[MAX_H][MAX_W][MAX_OC],
		int OC, int H_OUT, int W_OUT
)
{
	int errors = 0;
	int count = 0;


	for (int oc = 0; oc < OC; oc++) {
	    for (int h = 0; h < H_OUT; h++) {
	        for (int w = 0; w < W_OUT; w++) {
	        	if (count < 5){
	            	count++;
	            	std::cout << "Match: " << out_kernel[h][w][oc] << std::endl;
	            }

	            if (out_kernel[h][w][oc] != golden[h][w][oc]) {
	                errors++;
	                std::cout << "Mismatch @ (h=" << h
	                          << ",w=" << w << ",oc=" << oc << "): "
	                          << "Kernel=" << out_kernel[h][w][oc]
	                          << "  Ref=" << golden[h][w][oc]
	                          << std::endl;
	            }
	        }
	    }
	}

	// pass or fail
	if (errors == 0) {
	    std::cout << "Matches golden reference." << std::endl;
	} else {
	    std::cout << "Doesn't match golden reference: " << errors << " mismatches found." << std::endl;
	}
}


int main() {
    // test parameters
    int H = 32;
    int W = 32;
    int IC = 3;
    int OC = 96;
    int K = 3;
    int S = 2;
    int P = 1;
    
    static fixed_point_t activations[MAX_H][MAX_W][MAX_IC];
    static fixed_point_t weights[MAX_K][MAX_K][MAX_IC][MAX_OC];
    static fixed_point_t out_ws[MAX_H][MAX_W][MAX_OC];
    static fixed_point_t out_os[MAX_H][MAX_W][MAX_OC];
    static fixed_point_t golden[MAX_H][MAX_W][MAX_OC];

    // randomize inputs
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            for (int c = 0; c < IC; c++) {
            	float r = ((rand() % 256) / 16.0f) - 8.0f;
                activations[h][w][c] = (fixed_point_t)r;
            }
        }
    }

    for (int kh = 0; kh < K; kh++) {
        for (int kw = 0; kw < K; kw++) {
            for (int ic = 0; ic < IC; ic++) {
                for (int oc = 0; oc < OC; oc++) {
                	float r = ((rand() % 64) / 16.0f) - 2.0f;
                	weights[kh][kw][ic][oc] = (fixed_point_t)r;
                }
            }
        }
    }

    int H_OUT = (H + 2*P - K)/S + 1;
    int W_OUT = (W + 2*P - K)/S + 1;

    // compute golden result
    conv3d_golden(activations, weights, golden, H, W, IC, OC, K, S, P);

    // run kernel
    conv3d_ws(activations, weights, out_ws, H, W, IC, OC, K, S, P);
    // compare results
    compare_results(out_ws,golden,OC,H_OUT,W_OUT);

    // run kernel
    conv3d_os(activations, weights, out_os, H, W, IC, OC, K, S, P);
    compare_results(out_os,golden,OC,H_OUT,W_OUT);

    return 0;
}



