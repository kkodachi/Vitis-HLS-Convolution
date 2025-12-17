#include "config.h"
#include "kernel.h"
#include <cstdio>
#include <cstdint>
#include "weights.h"

// ============================================================================
// Helper functions to reshape flat arrays into multi-dimensional arrays
// ============================================================================

void load_conv1_weights(fixed_point_t dest[MAX_CONV_K][MAX_CONV_K][MAX_CONV1_IC][MAX_CONV1_OC]) {
    int idx = 0;
    for (int kh = 0; kh < 7; kh++) {
        for (int kw = 0; kw < 7; kw++) {
            for (int ic = 0; ic < 3; ic++) {
                for (int oc = 0; oc < 96; oc++) {
                    dest[kh][kw][ic][oc] = fixed_point_t(conv1_weights_flat[idx++]);
                }
            }
        }
    }
    // FILE* f = fopen("conv1_weights_flat.bin", "rb");
    // if (!f) {
    //     // printf("Error: cannot open %s\n", conv1_weights_flat.bin);
    //     return;
    // }

    // // Read the raw bytes into a flat temporary buffer
    // int8_t buffer[MAX_CONV_K * MAX_CONV_K * MAX_CONV1_IC * MAX_CONV1_OC];
    // size_t read = fread(buffer, sizeof(buffer[0]), MAX_CONV_K * MAX_CONV_K * MAX_CONV1_IC * MAX_CONV1_OC, f);
    // fclose(f);

    // // Copy into 4D fixed-point array
    // int idx = 0;
    // for (int kh = 0; kh < MAX_CONV_K; kh++) {
    //     for (int kw = 0; kw < MAX_CONV_K; kw++) {
    //         for (int ic = 0; ic < MAX_CONV1_IC; ic++) {
    //             for (int oc = 0; oc < MAX_CONV1_OC; oc++) {
    //                 #pragma HLS PIPELINE II=1
    //                 dest[kh][kw][ic][oc] = fixed_point_t(buffer[idx++]);
    //             }
    //         }
    //     }
    // }
}

void load_conv10_weights(fixed_point_t dest[MAX_FIRE_IC][AVGPOOL_C]) {
    int idx = 0;
    for (int ic = 0; ic < 512; ic++) {
        for (int oc = 0; oc < 10; oc++) {
            dest[ic][oc] = fixed_point_t(conv10_weights_flat[idx++]);
        }
    }
    // FILE* f = fopen("conv10_weights_flat.bin", "rb");
    // if (!f) {
    //     // printf("Error: cannot open %s\n", "conv10_weights_flat.bin");
    //     return;
    // }

    // // Read raw bytes into a flat buffer
    // int8_t buffer[MAX_FIRE_IC * AVGPOOL_C];
    // size_t read = fread(buffer, sizeof(buffer[0]), MAX_FIRE_IC * AVGPOOL_C, f);
    // // if (read != MAX_FIRE_IC * AVGPOOL_C) {
    //     // printf("Warning: only read %zu of %d weights\n", read, MAX_FIRE_IC * AVGPOOL_C);
    // // }
    // fclose(f);

    // // Copy into 2D fixed-point array
    // int idx = 0;
    // for (int ic = 0; ic < MAX_FIRE_IC; ic++) {
    //     for (int oc = 0; oc < AVGPOOL_C; oc++) {
    //         #pragma HLS PIPELINE II=1
    //         dest[ic][oc] = fixed_point_t(buffer[idx++]);
    //     }
    // }
}

void load_fire_squeeze_weights(const weight_t* src,int IC, int SC,
                                fixed_point_t dest[MAX_FIRE_IC][MAX_FIRE_SC],
                                const char* filename) {
    int idx = 0;
    for (int ic = 0; ic < IC; ic++) {
        for (int sc = 0; sc < SC; sc++) {
            dest[ic][sc] = fixed_point_t(src[idx++]);
        }
    }
    // FILE* f = fopen(filename, "rb");
    // if (!f) {
    //     printf("Error: cannot open %s\n", filename);
    //     return;
    // }

    // // Read raw bytes into a flat buffer
    // int8_t buffer[MAX_FIRE_IC * MAX_FIRE_SC];
    // size_t read = fread(buffer, sizeof(buffer[0]), IC * SC, f);
    // if (read != IC * SC) {
    //     printf("Warning: only read %zu of %d weights\n", read, IC * SC);
    // }
    // fclose(f);

    // // Copy into 2D fixed-point array
    // int idx = 0;
    // for (int ic = 0; ic < IC; ic++) {
    //     for (int sc = 0; sc < SC; sc++) {
    //         #pragma HLS PIPELINE II=1
    //         dest[ic][sc] = fixed_point_t(buffer[idx++]);
    //     }
    // }
}

void load_fire_expand1x1_weights(const weight_t* src,int SC, int EC,
                                  fixed_point_t dest[MAX_FIRE_SC][MAX_FIRE_EC],
                                    const char* filename) {
    int idx = 0;
    for (int sc = 0; sc < SC; sc++) {
        for (int ec = 0; ec < EC; ec++) {
            dest[sc][ec] = fixed_point_t(src[idx++]);
        }
    }

    // FILE* f = fopen(filename, "rb");
    // if (!f) {
    //     printf("Error: cannot open %s\n", filename);
    //     return;
    // }

    // // Read raw bytes into a flat buffer
    // int8_t buffer[MAX_FIRE_SC * MAX_FIRE_EC];
    // size_t read = fread(buffer, sizeof(buffer[0]), SC * EC, f);
    // if (read != SC * EC) {
    //     printf("Warning: only read %zu of %d weights\n", read, SC * EC);
    // }
    // fclose(f);

    // // Copy into 2D fixed-point array
    // int idx = 0;
    // for (int sc = 0; sc < SC; sc++) {
    //     for (int ec = 0; ec < EC; ec++) {
    //         #pragma HLS PIPELINE II=1
    //         dest[sc][ec] = fixed_point_t(buffer[idx++]);
    //     }
    // }
}

void load_fire_expand3x3_weights(const weight_t* src, int SC, int EC,
                                  fixed_point_t dest[3][3][MAX_FIRE_SC][MAX_FIRE_EC],
                                    const char* filename) {
    int idx = 0;
    for (int kh = 0; kh < 3; kh++) {
        for (int kw = 0; kw < 3; kw++) {
            for (int sc = 0; sc < SC; sc++) {
                for (int ec = 0; ec < EC; ec++) {
                    dest[kh][kw][sc][ec] = fixed_point_t(src[idx++]);
                }
            }
        }
    }
    // FILE* f = fopen(filename, "rb");
    // if (!f) {
    //     printf("Error: cannot open %s\n", filename);
    //     return;
    // }

    // // Read raw bytes into a flat buffer
    // int8_t buffer[3 * 3 * MAX_FIRE_SC * MAX_FIRE_EC];
    // size_t read = fread(buffer, sizeof(buffer[0]), 3 * 3 * SC * EC, f);
    // if (read != 3 * 3 * SC * EC) {
    //     printf("Warning: only read %zu of %d weights\n", read, 3 * 3 * SC * EC);
    // }
    // fclose(f);

    // // Copy into 4D fixed-point array
    // int idx = 0;
    // for (int kh = 0; kh < 3; kh++) {
    //     for (int kw = 0; kw < 3; kw++) {
    //         for (int sc = 0; sc < SC; sc++) {
    //             for (int ec = 0; ec < EC; ec++) {
    //                 #pragma HLS PIPELINE II=1
    //                 dest[kh][kw][sc][ec] = fixed_point_t(buffer[idx++]);
    //             }
    //         }
    //     }
    // }
}

// ============================================================================
// Static weight buffers - allocated once and reused
// ============================================================================

static fixed_point_t conv1_weights_buf[MAX_CONV_K][MAX_CONV_K][MAX_CONV1_IC][MAX_CONV1_OC];
static fixed_point_t conv10_weights_buf[MAX_FIRE_IC][AVGPOOL_C];

// Fire module weight buffers (reused for each fire layer)
static fixed_point_t fire_squeeze_buf[MAX_FIRE_IC][MAX_FIRE_SC];
static fixed_point_t fire_expand1x1_buf[MAX_FIRE_SC][MAX_FIRE_EC];
static fixed_point_t fire_expand3x3_buf[3][3][MAX_FIRE_SC][MAX_FIRE_EC];

// Track if weights have been loaded
static bool weights_initialized = false;

void initialize_weights() {
    if (weights_initialized) return;
    
    // Load Conv1 weights once
    load_conv1_weights(conv1_weights_buf);
    
    // Load Conv10 weights once
    load_conv10_weights(conv10_weights_buf);
    
    weights_initialized = true;
}

// ============================================================================
// Layer Configuration Function
// ============================================================================

void configure_layer(int layer, LayerConfig &config) {
    // Reset configuration
    config.layer_type = LAYER_NONE;
    config.H = 0;
    config.W = 0;
    config.IC = 0;
    config.OC = 0;
    config.K = 0;
    config.S = 0;
    config.P = 0;
    config.SC = 0;
    config.EC = 0;
    
    switch(layer) {
        case 0: // Conv1: 3->96, 224x224->112x112, K=7, S=2, P=3
            config.layer_type = LAYER_CONV;
            config.H = 224;
            config.W = 224;
            config.IC = 3;
            config.OC = 96;
            config.K = 7;
            config.S = 2;
            config.P = 3;
            break;
            
        case 1: // MaxPool1: 96 channels, 112x112->56x56, K=3, S=2
            config.layer_type = LAYER_MAXPOOL;
            config.H = 112;
            config.W = 112;
            config.IC = 96;
            break;
            
        case 2: // Fire2: 96->16->128, 56x56
            config.layer_type = LAYER_FIRE;
            config.H = 56;
            config.W = 56;
            config.IC = 96;
            config.SC = 16;
            config.EC = 64;
            config.fire_id = 2;
            break;
            
        case 3: // Fire3: 128->16->128, 56x56
            config.layer_type = LAYER_FIRE;
            config.H = 56;
            config.W = 56;
            config.IC = 128;
            config.SC = 16;
            config.EC = 64;
            config.fire_id = 3;
            break;
            
        case 4: // Fire4: 128->32->256, 56x56
            config.layer_type = LAYER_FIRE;
            config.H = 56;
            config.W = 56;
            config.IC = 128;
            config.SC = 32;
            config.EC = 128;
            config.fire_id = 4;
            break;
            
        case 5: // MaxPool2: 256 channels, 56x56->28x28, K=3, S=2
            config.layer_type = LAYER_MAXPOOL;
            config.H = 56;
            config.W = 56;
            config.IC = 256;
            break;
            
        case 6: // Fire5: 256->32->256, 28x28
            config.layer_type = LAYER_FIRE;
            config.H = 28;
            config.W = 28;
            config.IC = 256;
            config.SC = 32;
            config.EC = 128;
            config.fire_id = 5;
            break;
            
        case 7: // Fire6: 256->48->384, 28x28
            config.layer_type = LAYER_FIRE;
            config.H = 28;
            config.W = 28;
            config.IC = 256;
            config.SC = 48;
            config.EC = 192;
            config.fire_id = 6;
            break;
            
        case 8: // Fire7: 384->48->384, 28x28
            config.layer_type = LAYER_FIRE;
            config.H = 28;
            config.W = 28;
            config.IC = 384;
            config.SC = 48;
            config.EC = 192;
            config.fire_id = 7;
            break;
            
        case 9: // Fire8: 384->64->512, 28x28
            config.layer_type = LAYER_FIRE;
            config.H = 28;
            config.W = 28;
            config.IC = 384;
            config.SC = 64;
            config.EC = 256;
            config.fire_id = 8;
            break;
            
        case 10: // MaxPool3: 512 channels, 28x28->14x14, K=3, S=2
            config.layer_type = LAYER_MAXPOOL;
            config.H = 28;
            config.W = 28;
            config.IC = 512;
            break;
            
        case 11: // Fire9: 512->64->512, 14x14
            config.layer_type = LAYER_FIRE;
            config.H = 14;
            config.W = 14;
            config.IC = 512;
            config.SC = 64;
            config.EC = 256;
            config.fire_id = 9;
            break;
            
        case 12: // Conv10: 512->10, 14x14, K=1, S=1, P=0
            config.layer_type = LAYER_CONV10;
            config.H = 14;
            config.W = 14;
            config.IC = 512;
            config.OC = 10;
            config.K = 1;
            config.S = 1;
            config.P = 0;
            break;
            
        case 13: // AvgPool: 10 channels, 14x14->1x1
            config.layer_type = LAYER_AVGPOOL;
            config.H = 14;
            config.W = 14;
            config.IC = 10;
            break;
            
        default:
            config.layer_type = LAYER_NONE;
            break;
    }
}

// ============================================================================
// Load Fire Module Weights Based on Fire ID
// ============================================================================

void load_fire_weights(int fire_id) {
    switch(fire_id) {
        case 2:
            load_fire_squeeze_weights(fire2_squeeze_weights_flat,96, 16, fire_squeeze_buf,"fire2_squeeze_weights_flat.bin");
            load_fire_expand1x1_weights(fire2_expand1x1_weights_flat,16, 64, fire_expand1x1_buf,"fire2_expand1x1_weights_flat.bin");
            load_fire_expand3x3_weights(fire2_expand3x3_weights_flat,16, 64, fire_expand3x3_buf,"fire2_expand3x3_weights_flat.bin");
            break;
        case 3:
            load_fire_squeeze_weights(fire3_squeeze_weights_flat,128, 16, fire_squeeze_buf,"fire3_squeeze_weights_flat.bin");
            load_fire_expand1x1_weights(fire3_expand1x1_weights_flat,16, 64, fire_expand1x1_buf,"fire3_expand1x1_weights_flat.bin");
            load_fire_expand3x3_weights(fire3_expand3x3_weights_flat,16, 64, fire_expand3x3_buf,"fire3_expand3x3_weights_flat.bin");
            break;
        case 4:
            load_fire_squeeze_weights(fire4_squeeze_weights_flat,128, 32, fire_squeeze_buf,"fire4_squeeze_weights_flat.bin");
            load_fire_expand1x1_weights(fire4_expand1x1_weights_flat,32, 128, fire_expand1x1_buf,"fire4_expand1x1_weights_flat.bin");
            load_fire_expand3x3_weights(fire4_expand3x3_weights_flat,32, 128, fire_expand3x3_buf,"fire4_expand3x3_weights_flat.bin");
            break;
        case 5:
            load_fire_squeeze_weights(fire5_squeeze_weights_flat,256, 32, fire_squeeze_buf,"fire5_squeeze_weights_flat.bin");
            load_fire_expand1x1_weights(fire5_expand1x1_weights_flat,32, 128, fire_expand1x1_buf,"fire5_expand1x1_weights_flat.bin");
            load_fire_expand3x3_weights(fire5_expand3x3_weights_flat,32, 128, fire_expand3x3_buf,"fire5_expand3x3_weights_flat.bin");
            break;
        case 6:
            load_fire_squeeze_weights(fire6_squeeze_weights_flat,256, 48, fire_squeeze_buf,"fire6_squeeze_weights_flat.bin");
            load_fire_expand1x1_weights(fire6_expand1x1_weights_flat,48, 192, fire_expand1x1_buf,"fire6_expand1x1_weights_flat.bin");
            load_fire_expand3x3_weights(fire6_expand3x3_weights_flat,48, 192, fire_expand3x3_buf,"fire6_expand3x3_weights_flat.bin");
            break;
        case 7:
            load_fire_squeeze_weights(fire7_squeeze_weights_flat,384, 48, fire_squeeze_buf,"fire7_squeeze_weights_flat.bin");
            load_fire_expand1x1_weights(fire7_expand1x1_weights_flat,48, 192, fire_expand1x1_buf,"fire7_expand1x1_weights_flat.bin");
            load_fire_expand3x3_weights(fire7_expand3x3_weights_flat,48, 192, fire_expand3x3_buf,"fire7_expand3x3_weights_flat.bin");
            break;
        case 8:
            load_fire_squeeze_weights(fire8_squeeze_weights_flat,384, 64, fire_squeeze_buf,"fire8_squeeze_weights_flat.bin");
            load_fire_expand1x1_weights(fire8_expand1x1_weights_flat,64, 256, fire_expand1x1_buf,"fire8_expand1x1_weights_flat.bin");
            load_fire_expand3x3_weights(fire8_expand3x3_weights_flat,64, 256, fire_expand3x3_buf,"fire8_expand3x3_weights_flat.bin");
            break;
        case 9:
            load_fire_squeeze_weights(fire9_squeeze_weights_flat,512, 64, fire_squeeze_buf,"fire9_squeeze_weights_flat.bin");
            load_fire_expand1x1_weights(fire9_expand1x1_weights_flat,64, 256, fire_expand1x1_buf,"fire9_expand1x1_weights_flat.bin");
            load_fire_expand3x3_weights(fire9_expand3x3_weights_flat,64, 256, fire_expand3x3_buf,"fire9_expand3x3_weights_flat.bin");
            break;
    }
}

// ============================================================================
// Execute Layer
// ============================================================================

void execute_layer(
    LayerConfig &config,
    fixed_point_t input_buf[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],
    fixed_point_t output_buf[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC],
    fixed_point_t avgpool_output[AVGPOOL_C]
) {
    switch(config.layer_type) {
        case LAYER_CONV:
            // Conv1 layer - use conv1 not conv3d
            conv1(true, 
                  (fixed_point_t (*)[MAX_CONV_W][MAX_CONV1_IC])input_buf,
                  conv1_weights_buf,
                  output_buf,
                  config.H, config.W, config.IC, config.OC);
            break;
            
        case LAYER_MAXPOOL:
            maxpool(true, input_buf, output_buf, config.H, config.W, config.IC);
            break;
            
        case LAYER_FIRE:
            load_fire_weights(config.fire_id);
            fire(true, input_buf,
                 fire_squeeze_buf, fire_expand1x1_buf, fire_expand3x3_buf,
                 output_buf,
                 config.H, config.W, config.IC, config.SC, config.EC);
            break;
            
        case LAYER_CONV10:
            // Conv10 signature: conv10(bool, activations, weights, output)
            // No H, W, IC, OC parameters in your implementation
            conv10(true, input_buf, conv10_weights_buf,
                   (fixed_point_t (*)[AVGPOOL_W][AVGPOOL_C])output_buf);
            break;
            
        case LAYER_AVGPOOL:
            avgpool(true,
                    (fixed_point_t (*)[AVGPOOL_W][AVGPOOL_C])input_buf,
                    avgpool_output,
                    config.H, config.W, config.IC);
            break;
            
        case LAYER_NONE:
        default:
            // Do nothing
            break;
    }
}

// ============================================================================
// Main Controller Function
// ============================================================================

void squeezenet(
    fixed_point_t input_image[MAX_CONV_H * MAX_CONV_W *MAX_CONV1_IC],
    fixed_point_t final_output[AVGPOOL_C]
) {
    // ========================================================================
    // HLS Interface Pragmas
    // ========================================================================
    
    // AXI4-Master interface for input image (DDR memory access)
    #pragma HLS INTERFACE m_axi port=input_image offset=slave bundle=gmem0 depth=150528
    
    // AXI4-Master interface for output (DDR memory access)
    #pragma HLS INTERFACE m_axi port=final_output offset=slave bundle=gmem1 depth=10
    
    // AXI4-Lite slave interface for control signals
    #pragma HLS INTERFACE s_axilite port=input_image bundle=control
    #pragma HLS INTERFACE s_axilite port=final_output bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    
    // Initialize weights on first call
    initialize_weights();
    
    // Allocate ping-pong buffers
    static fixed_point_t buf1[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC];
    static fixed_point_t buf2[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC];
    
    // Copy input image to buf1 (first buffer)
    for (int h = 0; h < MAX_CONV_H; h++) {
        for (int w = 0; w < MAX_CONV_W; w++) {
            for (int c = 0; c < MAX_CONV1_IC; c++) {
                #pragma HLS PIPELINE II=1
                int idx = h * (MAX_CONV_W * MAX_CONV1_IC) + w * MAX_CONV1_IC + c;
                buf1[h][w][c] = input_image[idx];
            }
        }
    }
    
    // Ping-pong flag: true = buf1 is input, false = buf2 is input
    bool use_buf1_as_input = true;
    
    LayerConfig config;
    
    // Execute all 14 layers
    for (int layer = 0; layer < TOTAL_LAYERS; layer++) {
        // Get layer configuration
        configure_layer(layer, config);
        
        // Execute layer with appropriate buffers
        if (use_buf1_as_input) {
            execute_layer(config, buf1, buf2, final_output);
        } else {
            execute_layer(config, buf2, buf1, final_output);
        }
        
        // Toggle buffer for next layer
        use_buf1_as_input = !use_buf1_as_input;
    }
}