#include "config.h"
#include "kernel.h"

/*
look at squeezenet.ipynb and shapes.txt
imo just declare buf1 and buf2 in this file and reuse them
so first buf1 is input and buf2 is output then switch since each module feeds to the next
need to resize to smaller for avgpool but other than that every module should have input/output buffers same size
*/

void squeezenet(
    float input[MAX_CONV_H * MAX_CONV_H * 3], // 224x224x3 images
    int output[NUM_CLASSES]
)
{
    // convert flattened buffer to 3D
    fixed_point_t img[MAX_CONV_H][MAX_CONV_W][MAX_CONV_IC];

    // number of layers in model
    const int LAYERS = 14; // 15?
    // array of enable signals
    bool enables[MODULES] = {false,false,false,false};
    // array of args
    Args args[MODULES];

    for (int layer = 0; layer < LAYERS; layer++){
        controller(enables,layer,args);
        // conv3d(enables[CONV_IND],...)
        // maxpool(enables[MAXPOOL_IND],...)
        // fire(enables[FIRE_IND],...)
        // avgpool(enables[AVGPOOL_IND],...)
    }
}