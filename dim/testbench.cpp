#include "kernel.h"
#include "config.h"

#include <iostream>
#include <cstdlib>
#include <assert.h>
#include <cmath>

void avgpool_golden(
    bool enable,
    const fixed_point_t activations[AVGPOOL_H][AVGPOOL_W][AVGPOOL_C],
    fixed_point_t output[AVGPOOL_C],
    int H,      // input height
    int W,      // input width
    int IC     // input channels
)
{
    if (!enable) return;

    for (int ic = 0; ic < IC; ic++) {
        accum_t sum = 0;

        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                sum += activations[h][w][ic];
            }
        }
        output[ic] = sum / (H * W);
    }
}

void maxpool_golden(
    bool enable,
    const fixed_point_t activations[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],
    fixed_point_t output[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],
    int H,      // input height
    int W,      // input width
    int IC     // input channels
)
{
    if (!enable) return;

    const int K = 3;
    const int S = 2;
    const int P = 0;

    int H_OUT = (H + 2*P - K)/S + 1;
    int W_OUT = (W + 2*P - K)/S + 1;

    for (int ic = 0; ic < IC;ic++){
        for (int oh = 0; oh < H_OUT; oh++){
            for (int ow = 0; ow < W_OUT; ow++){
                fixed_point_t max_val = activations[oh*S][ow*S][ic];
                for (int kh = 0; kh < K; kh++) {
                    for (int kw = 0; kw < K; kw++) {
                        int ih = oh * S + kh;
                        int iw = ow * S + kw;

                        fixed_point_t v = activations[ih][iw][ic];
                        if (v > max_val)
                            max_val = v;
                    }
                }
                output[oh][ow][ic] = max_val;
            }
        }
    }
}


void conv3d_golden(
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
    int H_OUT = (H + 2*P - K) / S + 1;
    int W_OUT = (W + 2*P - K) / S + 1;

    assert(H_OUT <= MAX_FIRE_H);
    assert(W_OUT <= MAX_FIRE_W);
    assert(OC <= MAX_FIRE_IC);
    assert(IC <= MAX_FIRE_IC);

    for (int oc = 0; oc < OC; oc++) {
        for (int oh = 0; oh < H_OUT; oh++) {
            for (int ow = 0; ow < W_OUT; ow++) {
                
                accum_t sum = 0;

                for (int kh = 0; kh < K; kh++) {
                    for (int kw = 0; kw < K; kw++) {
                        for (int ic = 0; ic < IC; ic++) {

                            int ih = oh * S + kh - P;
                            int iw = ow * S + kw - P;

                            fixed_point_t val = 0;
                            // zero-padding
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                val = activations[ih][iw][ic];
                            }

                            sum += val * weights[kh][kw][ic][oc];
                        }
                    }
                }

                output[oh][ow][oc] = sum;
            }
        }
    }
}

void squeeze_golden(
    const fixed_point_t input[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],
    const fixed_point_t weights[MAX_FIRE_IC][MAX_FIRE_SC],
    fixed_point_t squeeze_output[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_SC],
    int H, int W, int IC, int SC
)
{
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            for (int sc = 0; sc < SC; sc++) {
                accum_t sum = 0;
                for (int ic = 0; ic < IC; ic++) {
                    sum += input[h][w][ic] * weights[ic][sc];
                }
                squeeze_output[h][w][sc] = (sum > 0) ? (fixed_point_t)sum : (fixed_point_t)0;
            }
        }
    }
}

void expand1_golden(
    const fixed_point_t input[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_SC],
    const fixed_point_t expand1x1_weights[MAX_FIRE_SC][MAX_FIRE_EC],
    fixed_point_t output[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],
    int H, int W, int SC, int EC
)
{
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            for (int ec = 0; ec < EC; ec++){    
                accum_t sum = 0;
                for (int sc = 0; sc < SC; sc++) {
                    sum += input[h][w][sc] * expand1x1_weights[sc][ec];
                }
                output[h][w][ec] = (sum > 0) ? (fixed_point_t)sum : (fixed_point_t)0;
            }
        }
    }
}

void expand3_golden(
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

    for (int ec = 0; ec < EC; ec++) {
        for (int h = 0; h < H_OUT; h++) {
            for (int w = 0; w < W_OUT; w++) {

                accum_t sum = 0;      // must accumulate across SC

                for (int sc = 0; sc < SC; sc++) {
                    for (int kh = 0; kh < K; kh++) {
                        for (int kw = 0; kw < K; kw++) {
                            int h_in = h * S + kh - P;
                            int w_in = w * S + kw - P;

                            fixed_point_t val = 0;
                            if (h_in >= 0 && h_in < H &&
                                w_in >= 0 && w_in < W)
                            {
                                val = input[h_in][w_in][sc];
                            }

                            sum += val * expand3x3_weights[kh][kw][sc][ec];
                        }
                    }
                }

                fixed_point_t relu = (sum > 0) ? (fixed_point_t)sum : (fixed_point_t)0;
                output[h][w][offset + ec] = relu;
            }
        }
    }

    // for (int ec = 0; ec < EC; ec++){
    //     for (int sc = 0; sc < SC; sc++){
    //         for (int h = 0; h < H_OUT; h++) {
    //             for (int w = 0; w < W_OUT; w++) {
    //                 accum_t sum = 0;
    //                 for (int kh = 0; kh < K; kh++) {
    //                     for (int kw = 0; kw < K; kw++) {
    //                         int h_in = h * S + kh - P;
    //                         int w_in = w * S + kw - P;

    //                         // zero padding
    //                         fixed_point_t val = 0;
    //                         if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
    //                             val = input[h_in][w_in][sc];
    //                         }

    //                         sum += val * expand3x3_weights[kh][kw][sc][ec];
    //                     }
    //                 }
    //                 output[h][w][offset+ec] = ((fixed_point_t)sum > 0) ? (fixed_point_t)sum : 0;
    //             }
    //         }
    //     }
    // }
}

void fire_golden(
    bool enable,
    const fixed_point_t input[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],
    const fixed_point_t squeeze_weights[MAX_FIRE_IC][MAX_FIRE_SC],
    const fixed_point_t expand1x1_weights[MAX_FIRE_SC][MAX_FIRE_EC],
    const fixed_point_t expand3x3_weights[3][3][MAX_FIRE_SC][MAX_FIRE_EC],
    fixed_point_t output[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],
    int H,
    int W,
    int IC,
    int SC, // squeeze channels
    int EC // expand channels
)
{
    if (!enable) return;

    fixed_point_t squeeze_output[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_SC];
    // #pragma HLS ARRAY_PARTITION variable=squeeze_output cyclic factor=8 dim=3

    // squeeze 1x1 conv + fused ReLU
    // squeeze(input,squeeze_weights,squeeze_output,H,W,IC,SC);
    squeeze_golden(input,squeeze_weights,squeeze_output,H,W,IC,SC);
    
    // expand 1x1 conv + fused ReLU, writes directly to output
    // expand1(squeeze_output,expand1x1_weights,output,H,W,SC,EC);
    expand1_golden(squeeze_output,expand1x1_weights,output,H,W,SC,EC);

    // squeeze 3x3 conv + fused ReLU, writes directly to output
    // expand3(squeeze_output,expand3x3_weights,output,H,W,SC,EC,EC);
    expand3_golden(squeeze_output,expand3x3_weights,output,H,W,SC,EC,EC);
}

// bool areFloatsEqual(float a, float b, float epsilon = 1e-6f) {
//     return std::fabs(a - b) < epsilon;
// }
bool areFloatsEqual(float a, float b, float epsilon = 1e-1f) {
    float diff = std::fabs(a - b);
    float maxVal = std::max(std::fabs(a), std::fabs(b));
    return diff <= epsilon * maxVal;
}

int main(){
    // Parameters below test conv1 -> maxpool1 -> fire2 in squeezenet.ipynb

    // CONV TEST START
    bool enable = true;

    int conv_H = 224;
    int conv_W = 224;
    int conv_IC = 3;

    int conv_OC = 96;
    int conv_K = 7;
    int conv_S = 2;
    int conv_P = 3;

    const int CONV_H_OUT = (conv_H + 2*conv_P - conv_K)/conv_S + 1; // 112
    const int CONV_W_OUT = (conv_W + 2*conv_P - conv_K)/conv_S + 1; // 112

    // std::cout << "CONV_H_OUT: " << CONV_H_OUT << " CONV_W_OUT: " << CONV_W_OUT << std::endl; 

    // fixed_point_t conv_in[MAX_CONV_H][MAX_CONV_W][MAX_CONV_IC];
    // fixed_point_t conv_w[MAX_CONV_K][MAX_CONV_K][MAX_CONV_IC][MAX_CONV_OC];
    // // fixed_point_t conv_out[MAX_CONV_H][MAX_CONV_W][MAX_CONV_OC];
    // // fixed_point_t conv_golden[MAX_CONV_H][MAX_CONV_W][MAX_CONV_OC];
    // fixed_point_t conv_out[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC];
    // fixed_point_t conv_golden[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC];

    auto conv_in = new fixed_point_t[MAX_CONV_H][MAX_CONV_W][MAX_CONV_IC];
    auto conv_w = new fixed_point_t[MAX_CONV_K][MAX_CONV_K][MAX_CONV_IC][MAX_CONV_OC];
    auto conv_out = new fixed_point_t[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC];
    auto conv_golden = new fixed_point_t[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC];
    
    // randomize input
    for (int h = 0; h < conv_H; h++){
        for (int w = 0; w < conv_W; w++){
            for (int c = 0; c < conv_IC; c++){
                int steps = std::rand() % 256;  // 8-bit range 0 to 255
                conv_in[h][w][c] = -8.0f + 0.0625f * steps;  // 2^-4 = 0.0625
            }
        }
    }

    // randomize weights
    for (int kh = 0; kh < conv_K; kh++){
        for (int kw = 0; kw < conv_K; kw++){
            for (int ic = 0; ic < conv_IC; ic++){
                for (int oc = 0; oc < conv_OC; oc++){
                    int steps = std::rand() % 256;  // 8-bit range 0 to 255
                    conv_w[kh][kw][ic][oc] = -8.0f + 0.0625f * steps;  // 2^-4 = 0.0625
                }
            }
        }
    }
    
    conv3d(enable,conv_in,conv_w,conv_out,conv_H,conv_W,conv_IC,conv_OC,conv_K,conv_S,conv_P);
    conv3d_golden(enable,conv_in,conv_w,conv_golden,conv_H,conv_W,conv_IC,conv_OC,conv_K,conv_S,conv_P);

    std::cout << "Comparing conv with golden" << std::endl;
    int count = 0;
    for (int h = 0; h < CONV_H_OUT; h++){
        for (int w = 0; w < CONV_W_OUT; w++){
            for (int oc = 0; oc < conv_OC; oc++){
                if (conv_out[h][w][oc] != conv_golden[h][w][oc]){
                    if (count < 5){
                        std::cout << "Mismatch at h, w, oc: " << h << " " << w << " " << oc << ", kernel: " << conv_out[h][w][oc] << " golden: " << conv_golden[h][w][oc] << std::endl;
                    }
                    count++;
                }
            }
        }
    }
    if (count != 0){
        std::cout << "conv3d() does not match golden: " << count << "/" << (CONV_H_OUT * CONV_W_OUT * conv_OC) << " mismatches" << std::endl;
    }
    else {
        std::cout << "conv3d() matches golden" << std::endl;
    }

    delete[] conv_in;
    delete[] conv_w;
    delete[] conv_golden;
    // CONV TEST END
    
    // MAXPOOL TEST START
    // fixed_point_t mp_out[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC];
    // fixed_point_t mp_golden[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC];
    auto mp_out = new fixed_point_t[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC];
    auto mp_golden = new fixed_point_t[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC];

    maxpool(enable,conv_out,mp_out,CONV_H_OUT,CONV_W_OUT,conv_OC);
    maxpool_golden(enable,conv_out,mp_golden,CONV_H_OUT,CONV_W_OUT,conv_OC);

    delete[] conv_out;

    const int MP_K = 3;
    const int MP_S = 2;
    const int MP_H_OUT = (CONV_H_OUT - MP_K) / MP_S + 1;
    const int MP_W_OUT = (CONV_W_OUT - MP_K) / MP_S + 1;
    const int MP_C_OUT = conv_OC;

    std::cout << "Comparing maxpool with golden" << std::endl;
    count = 0;
    for (int h = 0; h < MP_H_OUT; h++){
        for (int w = 0; w < MP_W_OUT; w++){
            for (int oc = 0; oc < MP_C_OUT; oc++){
                if (mp_out[h][w][oc] != mp_golden[h][w][oc]){
                    if (count < 5){
                        std::cout << "Mismatch at h, w, oc: " << h << " " << w << " " << oc << ", kernel: " << mp_out[h][w][oc] << " golden: " << mp_golden[h][w][oc] << std::endl;
                    }
                    count++;
                }
            }
        }
    }
    if (count != 0){
        std::cout << "maxpool() does not match golden: " << count << "/" << (MP_H_OUT * MP_W_OUT * conv_OC) << " mismatches" << std::endl;
    }
    else {
        std::cout << "maxpool() matches golden" << std::endl;
    }

    delete[] mp_golden;
    // MAXPOOL TEST END
    
    // FIRE TEST START
    // int FIRE_IC = 96;
    const int FIRE_IC = conv_OC;
    int FIRE_SC = 16;
    int FIRE_EC = 64;

    const int FIRE_OH = MP_H_OUT;
    const int FIRE_OW = MP_W_OUT;
    const int FIRE_OC = 2 * FIRE_EC;

    // fixed_point_t squeeze_weights[MAX_FIRE_IC][MAX_FIRE_SC];
    // fixed_point_t expand1x1_weights[MAX_FIRE_SC][MAX_FIRE_EC];
    // fixed_point_t expand3x3_weights[3][3][MAX_FIRE_SC][MAX_FIRE_EC];
    // fixed_point_t fire_out[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC];
    // fixed_point_t f_golden[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC];

    auto squeeze_weights = new fixed_point_t[MAX_FIRE_IC][MAX_FIRE_SC];
    auto expand1x1_weights = new fixed_point_t[MAX_FIRE_SC][MAX_FIRE_EC];
    auto expand3x3_weights = new fixed_point_t[3][3][MAX_FIRE_SC][MAX_FIRE_EC];
    auto fire_out = new fixed_point_t[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC];
    auto f_golden = new fixed_point_t[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC];

    // randomize squeeze weights
    for (int ic = 0; ic < FIRE_IC; ic++){
        for (int sc = 0; sc < FIRE_SC; sc++){
            int steps = std::rand() % 256;  // 8-bit range 0 to 255
            squeeze_weights[ic][sc] = -8.0f + 0.0625f * steps;  // 2^-4 = 0.0625
        }
    }

    // randomize expand weights
    for (int sc = 0; sc < FIRE_SC; sc++){
        for (int ec = 0; ec < FIRE_EC; ec++){
            int steps = std::rand() % 256;  // 8-bit range 0 to 255
            expand1x1_weights[sc][ec] = -8.0f + 0.0625f * steps;  // 2^-4 = 0.0625
            for (int i = 0; i < 3; i++){
                for (int j = 0; j < 3; j++){
                    steps = std::rand() % 256;  // 8-bit range 0 to 255
                    expand3x3_weights[i][j][sc][ec] = -8.0f + 0.0625f * steps;  // 2^-4 = 0.0625
                }
            }
        }
    }

    fire(enable,mp_out,squeeze_weights,expand1x1_weights,expand3x3_weights,fire_out,MP_H_OUT,MP_W_OUT,FIRE_IC,FIRE_SC,FIRE_EC);
    fire_golden(enable,mp_out,squeeze_weights,expand1x1_weights,expand3x3_weights,f_golden,MP_H_OUT,MP_W_OUT,FIRE_IC,FIRE_SC,FIRE_EC);

    delete[] mp_out;
    delete[] squeeze_weights;
    delete[] expand1x1_weights;
    delete[] expand3x3_weights;

    std::cout << "Comparing fire with golden" << std::endl;
    count = 0;
    for (int h = 0; h < FIRE_OH; h++){
        for (int w = 0; w < FIRE_OW; w++){
            for (int oc = 0; oc < FIRE_OC; oc++){
                // if (areFloatsEqual(fire_out[h][w][oc],f_golden[h][w][oc])){
                if (fire_out[h][w][oc] != f_golden[h][w][oc]){
                    if (count < 5){
                        std::cout << "Mismatch at h, w, oc: " << h << " " << w << " " << oc << ", kernel: " << fire_out[h][w][oc] << " golden: " << f_golden[h][w][oc] << std::endl;
                    }
                    count++;
                }
            }
        }
    }
    if (count != 0){
        std::cout << "fire() does not match golden: " << count << "/" << (FIRE_OH * FIRE_OW * FIRE_OC) << " mismatches" << std::endl;
    }
    else {
        std::cout << "fire() matches golden" << std::endl;
    }
    delete[] fire_out;
    delete[] f_golden;

    // // FIRE TEST END
    /*
    // AVGPOOL TEST START
    // test avgpool for final output

    fixed_point_t ap_in[AVGPOOL_H][AVGPOOL_W][AVGPOOL_C];

    fixed_point_t ap_out[AVGPOOL_C];
    fixed_point_t ap_golden[AVGPOOL_C];

    for (int h = 0; h < AVGPOOL_H; h++){
        for (int w = 0; w < AVGPOOL_W; w++){
            for (int c = 0; c < AVGPOOL_C; c++){
                int steps = std::rand() % 256;  // 8-bit range 0 to 255
                ap_in[h][w][c] = -8.0f + 0.0625f * steps;  // 2^-4 = 0.0625
            }
        }
    }

    avgpool(enable,ap_in,ap_out,AVGPOOL_H,AVGPOOL_W,AVGPOOL_C);
    avgpool_golden(enable,ap_in,ap_golden,AVGPOOL_H,AVGPOOL_W,AVGPOOL_C);

    std::cout << "Comparing avgpool with golden" << std::endl;
    count = 0;
    for (int c = 0; c < AVGPOOL_C; c++){
        if (ap_out[c] != ap_golden[c]){
            if (count < 5){
                std::cout << "Mismatch at c: " << c << ", kernel: " << ap_out[c] << " golden: " << ap_golden[c] << std::endl;
            }
            count++;
        }
    }
    if (count != 0){
        std::cout << "avgpool() does not match golden: " << count << "/" << AVGPOOL_C << " mismatches" << std::endl;
    }
    else {
        std::cout << "avgpool() matches golden" << std::endl;
    }
    // AVGPOOL TEST END
    */
}