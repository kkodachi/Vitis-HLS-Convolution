#include "config.h"
#include "conv3d_kernel.h"

inline size_t A_IDX_flat(size_t ci, size_t d, size_t h, size_t w,
                         size_t D_in, size_t H_in, size_t W_in) {
    return ((ci * D_in + d) * H_in + h) * W_in + w;
}
inline size_t OUT_IDX_flat(size_t oc, size_t od, size_t oh, size_t ow,
                           size_t D_out, size_t H_out, size_t W_out) {
    return ((oc * D_out + od) * H_out + oh) * W_out + ow;
}

void conv3d_ws(
    data_type *activations,
    size_t H_in, size_t W_in, size_t D_in,size_t C_in,
    data_type *weights,
    size_t Kh, size_t Kw, size_t Kd, size_t C_out,
    data_type *output,
    size_t stride_d, size_t stride_h, size_t stride_w,
    size_t pad_d, size_t pad_h, size_t pad_w
)
{
    #pragma HLS INTERFACE m_axi port=activations  offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=weights      offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=output       offset=slave bundle=gmem

    #pragma HLS INTERFACE s_axilite port=H_in      bundle=control
    #pragma HLS INTERFACE s_axilite port=W_in      bundle=control
    #pragma HLS INTERFACE s_axilite port=D_in      bundle=control
    #pragma HLS INTERFACE s_axilite port=C_in      bundle=control
    #pragma HLS INTERFACE s_axilite port=Kh        bundle=control
    #pragma HLS INTERFACE s_axilite port=Kw        bundle=control
    #pragma HLS INTERFACE s_axilite port=Kd        bundle=control
    #pragma HLS INTERFACE s_axilite port=C_out     bundle=control
    #pragma HLS INTERFACE s_axilite port=stride_d  bundle=control
    #pragma HLS INTERFACE s_axilite port=stride_h  bundle=control
    #pragma HLS INTERFACE s_axilite port=stride_w  bundle=control
    #pragma HLS INTERFACE s_axilite port=pad_d     bundle=control
    #pragma HLS INTERFACE s_axilite port=pad_h     bundle=control
    #pragma HLS INTERFACE s_axilite port=pad_w     bundle=control
    #pragma HLS INTERFACE s_axilite port=return     bundle=control

    // output dimensions
    size_t D_out = (D_in + 2*pad_d - Kd) / stride_d + 1;
    size_t H_out = (H_in + 2*pad_h - Kh) / stride_h + 1;
    size_t W_out = (W_in + 2*pad_w - Kw) / stride_w + 1;
    // number of weights
    // const size_t W_count = C_out * C_in * Kd * Kh * Kw;

    // weights for one channel
    const size_t WEIGHTS_PER_OC = C_in * Kd * Kh * Kw;
    // check if valid tile exists
    size_t max_tile_oc = BUFFER_SIZE / WEIGHTS_PER_OC; // integer division?
    // return if can't fit 1 channel into BRAM
    if (max_tile_oc == 0) {
        return;
    }

    data_type W_BRAM[BUFFER_SIZE];
    #pragma HLS BIND_STORAGE variable=W_BRAM type=ram_1p impl=bram
    data_type output_local[BUFFER_SIZE];
    #pragma HLS ARRAY_PARTITION variable=output_local complete dim=1

    // conv operation over each channel using the tiles

    // loop over tiles using max tile size
    for (size_t tile=0;tile<C_out;tile+=max_tile_oc){
        // current tile size
        size_t tile_oc = (tile + max_tile_oc <= C_out) ? max_tile_oc : (C_out - tile);
        // current tile weight count
        size_t tile_weight_count = tile_oc * WEIGHTS_PER_OC;
        // offset of tile to index DRAM
        size_t weights_oc_offset = tile * WEIGHTS_PER_OC;

        load_weights_tile:
        for (size_t i = 0; i < tile_weight_count; ++i) {
            #pragma HLS PIPELINE II=1
            W_BRAM[i] = weights[weights_oc_offset + i];
        }
        
        oc_tile_compute:
        for (size_t od=0;od<D_out;od++){
            for (size_t oh = 0; oh < H_out; oh++) {
                for (size_t ow = 0; ow < W_out; ow++) {
                    init_output_local:
                    for (size_t i=0;i<tile_oc;i++){
                        #pragma HLS UNROLL
                        output_local[i] = (data_type)0;
                    }
                    
                    for (size_t ic=0;ic<C_in;ic++){
                        for (size_t kd = 0; kd < Kd; ++kd) {
                            long in_d = (long)od * (long)stride_d - (long)pad_d + (long)kd;
                            if (in_d < 0 || in_d >= (long)D_in) continue; // account for padding

                            for (size_t kh = 0; kh < Kh; ++kh) {
                                long in_h = (long)oh * (long)stride_h - (long)pad_h + (long)kh;
                                if (in_h < 0 || in_h >= (long)H_in) continue; // account for padding

                                for (size_t kw = 0; kw < Kw; ++kw) {
                                    long in_w = (long)ow * (long)stride_w - (long)pad_w + (long)kw;
                                    if (in_w < 0 || in_w >= (long)W_in) continue; // account for padding

                                    // load activation once
                                    size_t a_idx = A_IDX_flat(ic, (size_t)in_d, (size_t)in_h, (size_t)in_w,
                                                             D_in, H_in, W_in);
                                    data_type a_val = activations[a_idx];

                                    // update all oc accumulators for this tile
                                    oc_update:
                                    for (size_t i = 0; i < tile_oc; ++i) {
                                        #pragma HLS PIPELINE II=1
                                        // compute local weight index
                                        size_t w_local_idx = i * WEIGHTS_PER_OC +
                                            (((ic * Kd + kd) * Kh + kh) * Kw + kw);
                                        data_type wval = W_BRAM[w_local_idx];

                                        output_local[i] += a_val * wval;
                                    }
                                }
                            }
                        }
                    }

                    for (size_t i = 0; i < tile_oc; ++i) {
                        #pragma HLS PIPELINE II=1
                        size_t oc_global = tile + i;
                        size_t out_idx = OUT_IDX_flat(oc_global, od, oh, ow, D_out, H_out, W_out);
                        output[out_idx] = output_local[i];
                    }

                }
            }
        }
        
    }
}
