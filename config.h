#ifndef CONFIG_H
#define CONFIG_H

#include "ap_fixed.h"

// Datatype used throughout
typedef ap_fixed<16,8> data_type;

// Maximum dimensions (required for HLS static arrays)
#define MAX_H   32
#define MAX_W   32
#define MAX_D   32

#define MAX_IC  16
#define MAX_OC  16

// Kernel size limits
#define MAX_KH  3
#define MAX_KW  3
#define MAX_KD  3

// BRAM buffer size for weight tiling
#define BUFFER_SIZE 32768

#endif




