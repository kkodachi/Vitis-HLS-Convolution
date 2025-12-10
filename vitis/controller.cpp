#include "kernel.h"

void controller(
    int stage,
    bool* en_conv,
    bool* en_maxpool,
    bool* en_avgpool,
    bool* en_relu,
    bool* en_fire
)
{
    #pragma HLS INLINE off
    
    // disable all by default
    *en_conv = false;
    *en_maxpool = false;
    *en_avgpool = false;
    *en_relu = false;
    *en_fire = false;

    // enable based on stage
    // SqueezeNet v1.0 architecture stages:
    // stage 0: conv1 (7x7, stride 2)
    // stage 1: relu
    // stage 2: maxpool
    // stage 3-10: fire modules (fire2-fire9)
    // stage 11: maxpool
    // stage 12: conv10 (1x1)
    // stage 13: relu
    // stage 14: avgpool (global)
    
    switch(stage) {
        case 0:  // conv1
            *en_conv = true;
            break;
        case 1:  // relu after conv1
            *en_relu = true;
            break;
        case 2:  // maxpool after conv1
            *en_maxpool = true;
            break;
        case 3:  // fire2
            *en_fire = true;
            break;
        case 4:  // fire3
            *en_fire = true;
            break;
        case 5:  // fire4
            *en_fire = true;
            break;
        case 6:  // maxpool
            *en_maxpool = true;
            break;
        case 7:  // fire5
            *en_fire = true;
            break;
        case 8:  // fire6
            *en_fire = true;
            break;
        case 9:  // fire7
            *en_fire = true;
            break;
        case 10: // fire8
            *en_fire = true;
            break;
        case 11: // maxpool
            *en_maxpool = true;
            break;
        case 12: // fire9
            *en_fire = true;
            break;
        case 13: // conv10
            *en_conv = true;
            break;
        case 14: // relu after conv10
            *en_relu = true;
            break;
        case 15: // avgpool (global)
            *en_avgpool = true;
            break;
        default:
            break;
    }
}
