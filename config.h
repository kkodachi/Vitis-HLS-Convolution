#ifndef CONFIG_H
#define CONFIG_H

#include "ap_fixed.h"

// Datatype used throughout
typedef ap_fixed<16,8> data_type;

// ULTRA-REDUCED dimensions for KV260
#define MAX_H   6   // Reduced from 8
#define MAX_W   6   // Reduced from 8
#define MAX_D   6   // Reduced from 8

#define MAX_IC  2   // Reduced from 4
#define MAX_OC  2   // Reduced from 4

// Kernel size limits
#define MAX_KH  3
#define MAX_KW  3
#define MAX_KD  3

#endif



