#include "config.h"
#include "kernel.h"

void controller(int layer, Args &args)
{
    // Reset all enable signals
    args.enable_conv = false;
    args.enable_maxpool = false;
    args.enable_fire = false;
    args.enable_avgpool = false;
    
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
            // Output: 112x112x96
            break;
            
        case 1: // MaxPool1: 96 channels, 112x112->56x56, K=3, S=2
            args.enable_maxpool = true;
            args.H = 112;
            args.W = 112;
            args.IC = 96;
            // Output: 56x56x96
            break;
            
        case 2: // Fire2: 96->16->128, 56x56
            args.enable_fire = true;
            args.H = 56;
            args.W = 56;
            args.IC = 96;
            args.SC = 16;
            args.EC = 64;
            // Output: 56x56x128 (64+64)
            break;
            
        case 3: // Fire3: 128->16->128, 56x56
            args.enable_fire = true;
            args.H = 56;
            args.W = 56;
            args.IC = 128;
            args.SC = 16;
            args.EC = 64;
            // Output: 56x56x128
            break;
            
        case 4: // Fire4: 128->32->256, 56x56
            args.enable_fire = true;
            args.H = 56;
            args.W = 56;
            args.IC = 128;
            args.SC = 32;
            args.EC = 128;
            // Output: 56x56x256
            break;
            
        case 5: // MaxPool2: 256 channels, 56x56->28x28, K=3, S=2
            args.enable_maxpool = true;
            args.H = 56;
            args.W = 56;
            args.IC = 256;
            // Output: 28x28x256
            break;
            
        case 6: // Fire5: 256->32->256, 28x28
            args.enable_fire = true;
            args.H = 28;
            args.W = 28;
            args.IC = 256;
            args.SC = 32;
            args.EC = 128;
            // Output: 28x28x256
            break;
            
        case 7: // Fire6: 256->48->384, 28x28
            args.enable_fire = true;
            args.H = 28;
            args.W = 28;
            args.IC = 256;
            args.SC = 48;
            args.EC = 192;
            // Output: 28x28x384
            break;
            
        case 8: // Fire7: 384->48->384, 28x28
            args.enable_fire = true;
            args.H = 28;
            args.W = 28;
            args.IC = 384;
            args.SC = 48;
            args.EC = 192;
            // Output: 28x28x384
            break;
            
        case 9: // Fire8: 384->64->512, 28x28
            args.enable_fire = true;
            args.H = 28;
            args.W = 28;
            args.IC = 384;
            args.SC = 64;
            args.EC = 256;
            // Output: 28x28x512
            break;
            
        case 10: // MaxPool3: 512 channels, 28x28->14x14, K=3, S=2
            args.enable_maxpool = true;
            args.H = 28;
            args.W = 28;
            args.IC = 512;
            // Output: 14x14x512
            break;
            
        case 11: // Fire9: 512->64->512, 14x14
            args.enable_fire = true;
            args.H = 14;
            args.W = 14;
            args.IC = 512;
            args.SC = 64;
            args.EC = 256;
            // Output: 14x14x512
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
            // Output: 14x14x10
            break;
            
        case 13: // AvgPool: 10 channels, 14x14->1x1
            args.enable_avgpool = true;
            args.H = 14;
            args.W = 14;
            args.IC = 10;
            // Output: 10
            break;
            
        default:
            // Invalid layer - all enables remain false
            break;
    }
}