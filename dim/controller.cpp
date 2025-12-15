#include "config.h"
#include "kernel.h"
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
                    dest[kh][kw][ic][oc] = conv1_weights_flat[idx++];
                }
            }
        }
    }
}

void load_conv10_weights(fixed_point_t dest[MAX_FIRE_IC][AVGPOOL_C]) {
    int idx = 0;
    for (int ic = 0; ic < 512; ic++) {
        for (int oc = 0; oc < 10; oc++) {
            dest[ic][oc] = conv10_weights_flat[idx++];
        }
    }
}

void load_fire_squeeze_weights(const fixed_point_t* src, int IC, int SC, 
                                fixed_point_t dest[MAX_FIRE_IC][MAX_FIRE_SC]) {
    int idx = 0;
    for (int ic = 0; ic < IC; ic++) {
        for (int sc = 0; sc < SC; sc++) {
            dest[ic][sc] = src[idx++];
        }
    }
}

void load_fire_expand1x1_weights(const fixed_point_t* src, int SC, int EC,
                                  fixed_point_t dest[MAX_FIRE_SC][MAX_FIRE_EC]) {
    int idx = 0;
    for (int sc = 0; sc < SC; sc++) {
        for (int ec = 0; ec < EC; ec++) {
            dest[sc][ec] = src[idx++];
        }
    }
}

void load_fire_expand3x3_weights(const fixed_point_t* src, int SC, int EC,
                                  fixed_point_t dest[3][3][MAX_FIRE_SC][MAX_FIRE_EC]) {
    int idx = 0;
    for (int kh = 0; kh < 3; kh++) {
        for (int kw = 0; kw < 3; kw++) {
            for (int sc = 0; sc < SC; sc++) {
                for (int ec = 0; ec < EC; ec++) {
                    dest[kh][kw][sc][ec] = src[idx++];
                }
            }
        }
    }
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
// Controller Function - Configures layer parameters and loads weights
// ============================================================================

void controller(int layer, Args &args)
{
    // Initialize weights on first call
    initialize_weights();
    
    // Reset all enable signals
    args.enable_conv = false;
    args.enable_maxpool = false;
    args.enable_fire = false;
    args.enable_avgpool = false;
    
    // Reset weight pointers
    args.conv1_weights = nullptr;
    args.conv10_weights = nullptr;
    args.squeeze_weights = nullptr;
    args.expand1x1_weights = nullptr;
    args.expand3x3_weights = nullptr;
    
    switch(layer) {
        case 0: // Conv1: 3->96, 224x224->112x112, K=7, S=2, P=3
            args.enable_conv = true;
            args.H = 224;
            args.W = 224;
            args.IC = 3;
            args.OC = 96;
            args.K = 7;
            args.S = 2;
            args.P = 3;
            args.conv1_weights = conv1_weights_buf;
            break;
            
        case 1: // MaxPool1: 96 channels, 112x112->56x56, K=3, S=2
            args.enable_maxpool = true;
            args.H = 112;
            args.W = 112;
            args.IC = 96;
            break;
            
        case 2: // Fire2: 96->16->128, 56x56
            args.enable_fire = true;
            args.H = 56;
            args.W = 56;
            args.IC = 96;
            args.SC = 16;
            args.EC = 64;
            load_fire_squeeze_weights(fire2_squeeze_weights_flat, 96, 16, fire_squeeze_buf);
            load_fire_expand1x1_weights(fire2_expand1x1_weights_flat, 16, 64, fire_expand1x1_buf);
            load_fire_expand3x3_weights(fire2_expand3x3_weights_flat, 16, 64, fire_expand3x3_buf);
            args.squeeze_weights = fire_squeeze_buf;
            args.expand1x1_weights = fire_expand1x1_buf;
            args.expand3x3_weights = fire_expand3x3_buf;
            break;
            
        case 3: // Fire3: 128->16->128, 56x56
            args.enable_fire = true;
            args.H = 56;
            args.W = 56;
            args.IC = 128;
            args.SC = 16;
            args.EC = 64;
            load_fire_squeeze_weights(fire3_squeeze_weights_flat, 128, 16, fire_squeeze_buf);
            load_fire_expand1x1_weights(fire3_expand1x1_weights_flat, 16, 64, fire_expand1x1_buf);
            load_fire_expand3x3_weights(fire3_expand3x3_weights_flat, 16, 64, fire_expand3x3_buf);
            args.squeeze_weights = fire_squeeze_buf;
            args.expand1x1_weights = fire_expand1x1_buf;
            args.expand3x3_weights = fire_expand3x3_buf;
            break;
            
        case 4: // Fire4: 128->32->256, 56x56
            args.enable_fire = true;
            args.H = 56;
            args.W = 56;
            args.IC = 128;
            args.SC = 32;
            args.EC = 128;
            load_fire_squeeze_weights(fire4_squeeze_weights_flat, 128, 32, fire_squeeze_buf);
            load_fire_expand1x1_weights(fire4_expand1x1_weights_flat, 32, 128, fire_expand1x1_buf);
            load_fire_expand3x3_weights(fire4_expand3x3_weights_flat, 32, 128, fire_expand3x3_buf);
            args.squeeze_weights = fire_squeeze_buf;
            args.expand1x1_weights = fire_expand1x1_buf;
            args.expand3x3_weights = fire_expand3x3_buf;
            break;
            
        case 5: // MaxPool2: 256 channels, 56x56->28x28, K=3, S=2
            args.enable_maxpool = true;
            args.H = 56;
            args.W = 56;
            args.IC = 256;
            break;
            
        case 6: // Fire5: 256->32->256, 28x28
            args.enable_fire = true;
            args.H = 28;
            args.W = 28;
            args.IC = 256;
            args.SC = 32;
            args.EC = 128;
            load_fire_squeeze_weights(fire5_squeeze_weights_flat, 256, 32, fire_squeeze_buf);
            load_fire_expand1x1_weights(fire5_expand1x1_weights_flat, 32, 128, fire_expand1x1_buf);
            load_fire_expand3x3_weights(fire5_expand3x3_weights_flat, 32, 128, fire_expand3x3_buf);
            args.squeeze_weights = fire_squeeze_buf;
            args.expand1x1_weights = fire_expand1x1_buf;
            args.expand3x3_weights = fire_expand3x3_buf;
            break;
            
        case 7: // Fire6: 256->48->384, 28x28
            args.enable_fire = true;
            args.H = 28;
            args.W = 28;
            args.IC = 256;
            args.SC = 48;
            args.EC = 192;
            load_fire_squeeze_weights(fire6_squeeze_weights_flat, 256, 48, fire_squeeze_buf);
            load_fire_expand1x1_weights(fire6_expand1x1_weights_flat, 48, 192, fire_expand1x1_buf);
            load_fire_expand3x3_weights(fire6_expand3x3_weights_flat, 48, 192, fire_expand3x3_buf);
            args.squeeze_weights = fire_squeeze_buf;
            args.expand1x1_weights = fire_expand1x1_buf;
            args.expand3x3_weights = fire_expand3x3_buf;
            break;
            
        case 8: // Fire7: 384->48->384, 28x28
            args.enable_fire = true;
            args.H = 28;
            args.W = 28;
            args.IC = 384;
            args.SC = 48;
            args.EC = 192;
            load_fire_squeeze_weights(fire7_squeeze_weights_flat, 384, 48, fire_squeeze_buf);
            load_fire_expand1x1_weights(fire7_expand1x1_weights_flat, 48, 192, fire_expand1x1_buf);
            load_fire_expand3x3_weights(fire7_expand3x3_weights_flat, 48, 192, fire_expand3x3_buf);
            args.squeeze_weights = fire_squeeze_buf;
            args.expand1x1_weights = fire_expand1x1_buf;
            args.expand3x3_weights = fire_expand3x3_buf;
            break;
            
        case 9: // Fire8: 384->64->512, 28x28
            args.enable_fire = true;
            args.H = 28;
            args.W = 28;
            args.IC = 384;
            args.SC = 64;
            args.EC = 256;
            load_fire_squeeze_weights(fire8_squeeze_weights_flat, 384, 64, fire_squeeze_buf);
            load_fire_expand1x1_weights(fire8_expand1x1_weights_flat, 64, 256, fire_expand1x1_buf);
            load_fire_expand3x3_weights(fire8_expand3x3_weights_flat, 64, 256, fire_expand3x3_buf);
            args.squeeze_weights = fire_squeeze_buf;
            args.expand1x1_weights = fire_expand1x1_buf;
            args.expand3x3_weights = fire_expand3x3_buf;
            break;
            
        case 10: // MaxPool3: 512 channels, 28x28->14x14, K=3, S=2
            args.enable_maxpool = true;
            args.H = 28;
            args.W = 28;
            args.IC = 512;
            break;
            
        case 11: // Fire9: 512->64->512, 14x14
            args.enable_fire = true;
            args.H = 14;
            args.W = 14;
            args.IC = 512;
            args.SC = 64;
            args.EC = 256;
            load_fire_squeeze_weights(fire9_squeeze_weights_flat, 512, 64, fire_squeeze_buf);
            load_fire_expand1x1_weights(fire9_expand1x1_weights_flat, 64, 256, fire_expand1x1_buf);
            load_fire_expand3x3_weights(fire9_expand3x3_weights_flat, 64, 256, fire_expand3x3_buf);
            args.squeeze_weights = fire_squeeze_buf;
            args.expand1x1_weights = fire_expand1x1_buf;
            args.expand3x3_weights = fire_expand3x3_buf;
            break;
            
        case 12: // Conv10: 512->10, 14x14, K=1, S=1, P=0
            args.enable_conv = true;
            args.H = 14;
            args.W = 14;
            args.IC = 512;
            args.OC = 10;
            args.K = 1;
            args.S = 1;
            args.P = 0;
            args.conv10_weights = conv10_weights_buf;
            break;
            
        case 13: // AvgPool: 10 channels, 14x14->1x1
            args.enable_avgpool = true;
            args.H = 14;
            args.W = 14;
            args.IC = 10;
            break;
            
        default:
            // Invalid layer - all enables remain false
            break;
    }
}