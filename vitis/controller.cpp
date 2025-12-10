#include "kernel.h"

void controller(
    int stage,
    bool* en_conv,
    bool* en_maxpool,
    bool* en_avgpool,
    bool* en_relu,
    bool* en_fire,
    int* H_in,
    int* W_in,
    int* IC_in,
    int* H_out,
    int* W_out,
    int* OC_out,
    int* K,
    int* stride,
    int* pad,
    int* squeeze_ch,
    int* expand_ch,
    int* weight_offset
)
{
    #pragma HLS INLINE off
    
    // disable all by default
    *en_conv = false;
    *en_maxpool = false;
    *en_avgpool = false;
    *en_relu = false;
    *en_fire = false;

    // SqueezeNet v1.0 architecture for CIFAR-10
    // Stage-by-stage dimension tracking
    
    switch(stage) {
        case 0:  // conv1: 32x32x3 -> 32x32x64 (3x3, stride=1, pad=1)
            *en_conv = true;
            *H_in = 32; *W_in = 32; *IC_in = 3;
            *K = 3; *stride = 1; *pad = 1;
            *OC_out = 64;
            *H_out = (*H_in + 2*(*pad) - *K) / *stride + 1; // 32
            *W_out = (*W_in + 2*(*pad) - *K) / *stride + 1; // 32
            *weight_offset = 0; // 3*3*3*64 = 1728
            break;
            
        case 1:  // relu: 32x32x64 -> 32x32x64 (in-place)
            *en_relu = true;
            *H_in = 32; *W_in = 32; *IC_in = 64;
            *H_out = 32; *W_out = 32; *OC_out = 64;
            break;
            
        case 2:  // maxpool: 32x32x64 -> 16x16x64 (2x2, stride=2)
            *en_maxpool = true;
            *H_in = 32; *W_in = 32; *IC_in = 64;
            *K = 2; *stride = 2;
            *H_out = (*H_in - *K) / *stride + 1; // 16
            *W_out = (*W_in - *K) / *stride + 1; // 16
            *OC_out = 64;
            break;
            
        case 3:  // fire2: 16x16x64 -> 16x16x128 (sq=16, ex=64)
            *en_fire = true;
            *H_in = 16; *W_in = 16; *IC_in = 64;
            *squeeze_ch = 16;
            *expand_ch = 64;
            *H_out = 16; *W_out = 16;
            *OC_out = *expand_ch * 2; // 128
            *weight_offset = 1728;
            break;
            
        case 4:  // fire3: 16x16x128 -> 16x16x128 (sq=16, ex=64)
            *en_fire = true;
            *H_in = 16; *W_in = 16; *IC_in = 128;
            *squeeze_ch = 16;
            *expand_ch = 64;
            *H_out = 16; *W_out = 16;
            *OC_out = *expand_ch * 2; // 128
            *weight_offset = 1728 + 11264;
            break;
            
        case 5:  // fire4: 16x16x128 -> 16x16x256 (sq=32, ex=128)
            *en_fire = true;
            *H_in = 16; *W_in = 16; *IC_in = 128;
            *squeeze_ch = 32;
            *expand_ch = 128;
            *H_out = 16; *W_out = 16;
            *OC_out = *expand_ch * 2; // 256
            *weight_offset = 1728 + 11264 + 12288;
            break;
            
        case 6:  // maxpool: 16x16x256 -> 8x8x256 (2x2, stride=2)
            *en_maxpool = true;
            *H_in = 16; *W_in = 16; *IC_in = 256;
            *K = 2; *stride = 2;
            *H_out = (*H_in - *K) / *stride + 1; // 8
            *W_out = (*W_in - *K) / *stride + 1; // 8
            *OC_out = 256;
            break;
            
        case 7:  // fire5: 8x8x256 -> 8x8x256 (sq=32, ex=128)
            *en_fire = true;
            *H_in = 8; *W_in = 8; *IC_in = 256;
            *squeeze_ch = 32;
            *expand_ch = 128;
            *H_out = 8; *W_out = 8;
            *OC_out = *expand_ch * 2; // 256
            *weight_offset = 1728 + 11264 + 12288 + 45056;
            break;
            
        case 8:  // fire6: 8x8x256 -> 8x8x384 (sq=48, ex=192)
            *en_fire = true;
            *H_in = 8; *W_in = 8; *IC_in = 256;
            *squeeze_ch = 48;
            *expand_ch = 192;
            *H_out = 8; *W_out = 8;
            *OC_out = *expand_ch * 2; // 384
            *weight_offset = 1728 + 11264 + 12288 + 45056 + 49152;
            break;
            
        case 9:  // fire7: 8x8x384 -> 8x8x384 (sq=48, ex=192)
            *en_fire = true;
            *H_in = 8; *W_in = 8; *IC_in = 384;
            *squeeze_ch = 48;
            *expand_ch = 192;
            *H_out = 8; *W_out = 8;
            *OC_out = *expand_ch * 2; // 384
            *weight_offset = 1728 + 11264 + 12288 + 45056 + 49152 + 104448;
            break;
            
        case 10: // fire8: 8x8x384 -> 8x8x512 (sq=64, ex=256)
            *en_fire = true;
            *H_in = 8; *W_in = 8; *IC_in = 384;
            *squeeze_ch = 64;
            *expand_ch = 256;
            *H_out = 8; *W_out = 8;
            *OC_out = *expand_ch * 2; // 512
            *weight_offset = 1728 + 11264 + 12288 + 45056 + 49152 + 104448 + 110592;
            break;
            
        case 11: // maxpool: 8x8x512 -> 4x4x512 (2x2, stride=2)
            *en_maxpool = true;
            *H_in = 8; *W_in = 8; *IC_in = 512;
            *K = 2; *stride = 2;
            *H_out = (*H_in - *K) / *stride + 1; // 4
            *W_out = (*W_in - *K) / *stride + 1; // 4
            *OC_out = 512;
            break;
            
        case 12: // fire9: 4x4x512 -> 4x4x512 (sq=64, ex=256)
            *en_fire = true;
            *H_in = 4; *W_in = 4; *IC_in = 512;
            *squeeze_ch = 64;
            *expand_ch = 256;
            *H_out = 4; *W_out = 4;
            *OC_out = *expand_ch * 2; // 512
            *weight_offset = 1728 + 11264 + 12288 + 45056 + 49152 + 104448 + 110592 + 188416;
            break;
            
        case 13: // conv10: 4x4x512 -> 4x4x10 (1x1, stride=1, pad=0)
            *en_conv = true;
            *H_in = 4; *W_in = 4; *IC_in = 512;
            *OC_out = 10;
            *K = 1; *stride = 1; *pad = 0;
            *H_out = (*H_in + 2*(*pad) - *K) / *stride + 1; // 4
            *W_out = (*W_in + 2*(*pad) - *K) / *stride + 1; // 4
            *weight_offset = 1728 + 11264 + 12288 + 45056 + 49152 + 104448 + 110592 + 188416 + 196608;
            break;
            
        case 14: // relu: 4x4x10 -> 4x4x10 (in-place)
            *en_relu = true;
            *H_in = 4; *W_in = 4; *IC_in = 10;
            *H_out = 4; *W_out = 4; *OC_out = 10;
            break;
            
        case 15: // avgpool: 4x4x10 -> 1x1x10 (global, 4x4 kernel)
            *en_avgpool = true;
            *H_in = 4; *W_in = 4; *IC_in = 10;
            *K = 4; *stride = 1;
            *H_out = (*H_in - *K) / *stride + 1; // 1
            *W_out = (*W_in - *K) / *stride + 1; // 1
            *OC_out = 10;
            break;
            
        default:
            break;
    }
}