#include "config.h"
#include "kernel.h"

void load_conv_weights(
    const fixed_point_t* flat_weights,        // linear array [K*K*IC*OC]
    fixed_point_t dest[MAX_CONV_K][MAX_CONV_K][MAX_CONV_DIM][MAX_CONV_DIM],
    int K, int IC, int OC
) {
    int idx = 0;
    for (int kh = 0; kh < K; kh++)
        for (int kw = 0; kw < K; kw++)
            for (int ic = 0; ic < IC; ic++)
                for (int oc = 0; oc < OC; oc++)
                    #pragma HLS PIPELINE II=1
                    dest[kh][kw][ic][oc] = (fixed_point_t)flat_weights[idx++];
}

void squeezenet(
    fixed_point_t input[CONV1_DIM * CONV1_DIM * CONV1_IC],
    fixed_point_t* output,
    fixed_point_t conv1w[MAX_CONV_K * MAX_CONV_K * CONV1_IC * CONV1_OC],
    fixed_point_t conv2w[MAX_CONV_K * MAX_CONV_K * CONV2_IC *CONV2_OC],
    fixed_point_t conv3w[MAX_CONV_K * MAX_CONV_K * CONV3_IC *CONV3_OC],
    fixed_point_t conv4w[MAX_CONV_K * MAX_CONV_K * CONV4_IC *CONV4_OC],
    fixed_point_t conv5w[CONV1_IC * NUM_CLASSES]
)
{
    const int LAYERS = 4;

    int H = CONV1_DIM;
    int W = CONV1_DIM;
    int IC = CONV1_IC;
    int OC = CONV1_OC;

    fixed_point_t buf1[MAX_CONV_DIM][MAX_CONV_DIM][MAX_CONV_DIM];
    fixed_point_t buf2[MAX_CONV_DIM][MAX_CONV_DIM][MAX_CONV_DIM];
    fixed_point_t weights[MAX_CONV_K][MAX_CONV_K][MAX_CONV_DIM][MAX_CONV_DIM];

    fixed_point_t (*in_buf)[MAX_CONV_DIM][MAX_CONV_DIM]  = buf1;
    fixed_point_t (*out_buf)[MAX_CONV_DIM][MAX_CONV_DIM] = buf2;
    fixed_point_t* flat_weights;

    for (int layer = 0; layer < LAYERS; layer++){
        switch(layer){
            case 0:
                H = CONV1_DIM;
                W = CONV1_DIM;
                IC = CONV1_IC;
                OC = CONV1_OC;
                flat_weights = conv1w;
                break;
            case 1: // load CONV2
                H = CONV2_DIM;
                W = CONV2_DIM;
                IC = CONV2_IC;
                OC = CONV2_OC;
                flat_weights = conv2w;
                break;
            case 2: // load CONV3
                H = CONV3_DIM;
                W = CONV3_DIM;
                IC = CONV3_IC;
                OC = CONV3_OC;
                flat_weights = conv3w;
                break;
            case 3: // load CONV4
                H = CONV4_DIM;
                W = CONV4_DIM;
                IC = CONV4_IC;
                OC = CONV4_OC;
                flat_weights = conv4w;
                break;
        }
        load_conv_weights(flat_weights,weights,MAX_CONV_K,IC,OC);
        conv(in_buf,weights,out_buf,H,W,IC,OC);

        fixed_point_t (*tmp)[MAX_CONV_DIM][MAX_CONV_DIM] = in_buf;
        in_buf  = out_buf;
        out_buf = tmp;
    }
    H = CONV5_DIM;
    W = CONV5_DIM;
    IC = CONV5_IC;
    OC = NUM_CLASSES;

    fixed_point_t conv5_out[CONV5_DIM][CONV5_DIM][NUM_CLASSES];
    fixed_point_t weights5[CONV4_OC][NUM_CLASSES];

    // load weights5
    for (int ic = 0; ic < CONV4_OC; ic++) {
        for (int oc = 0; oc < NUM_CLASSES; oc++) {
            #pragma HLS PIPELINE II=1
            int idx = ic * NUM_CLASSES + oc;
            weights5[ic][oc] = conv5w[idx];
        }
    }
    // call conv5
    conv5(in_buf,weights5,conv5_out,H,W,IC,OC);

    // call avgpool
    IC = NUM_CLASSES;
    fixed_point_t final_output[NUM_CLASSES];
    avgpool(conv5_out,final_output,H,W,IC);

    for (int i = 0; i < NUM_CLASSES; i++) {
        #pragma HLS PIPELINE II=1
        output[i] = final_output[i];
    }
}