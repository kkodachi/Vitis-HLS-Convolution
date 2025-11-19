#ifndef CONFIG_H
#define CONFIG_H

#include "ap_fixed.h"

// Datatype used throughout
typedef ap_fixed<16,8> data_type;

// REDUCED Maximum dimensions to fit on KV260
#define MAX_H   8   // Reduced from 32
#define MAX_W   8   // Reduced from 32
#define MAX_D   8   // Reduced from 32

#define MAX_IC  4   // Reduced from 16
#define MAX_OC  4   // Reduced from 16

// Kernel size limits
#define MAX_KH  3
#define MAX_KW  3
#define MAX_KD  3

#endif



