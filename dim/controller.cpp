#include "config.h"
#include "kernel.h"

void controller(bool enables[],int layer, Args args[])
{
    switch(layer) {
        case 0:
            enables[CONV_IND] = true;
            enables[MAXPOOL_IND] = false;
            enables[FIRE_IND] = false;
            enables[AVGPOOL_IND] = false;
            break;

        default:
            // throw error or something
    }
}