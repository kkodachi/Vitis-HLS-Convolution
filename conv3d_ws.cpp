#include "conv3d_kernel.h"
#include "config.h"

void conv3d_ws(
	data_type *activations, size_t H, size_t W, size_t D, size_t IC,
	data_type *weights, size_t Kh, size_t Kw, size_t Kd, size_t OC,
	data_type *output_ws,
	size_t stride_d, size_t stride_h, size_t stride_w,
	size_t pad_d, size_t pad_h, size_t pad_w
) {
	// Minimal AXI interface - single port, minimal burst
	#pragma HLS INTERFACE m_axi port=activations depth=250 bundle=gmem latency=64
	#pragma HLS INTERFACE m_axi port=weights depth=250 bundle=gmem latency=64
	#pragma HLS INTERFACE m_axi port=output_ws depth=250 bundle=gmem latency=64

	#pragma HLS INTERFACE s_axilite port=activations bundle=control
	#pragma HLS INTERFACE s_axilite port=weights bundle=control
	#pragma HLS INTERFACE s_axilite port=output_ws bundle=control

	#pragma HLS INTERFACE s_axilite port=H bundle=control
	#pragma HLS INTERFACE s_axilite port=W bundle=control
	#pragma HLS INTERFACE s_axilite port=D bundle=control
	#pragma HLS INTERFACE s_axilite port=IC bundle=control

	#pragma HLS INTERFACE s_axilite port=Kh bundle=control
	#pragma HLS INTERFACE s_axilite port=Kw bundle=control
	#pragma HLS INTERFACE s_axilite port=Kd bundle=control
	#pragma HLS INTERFACE s_axilite port=OC bundle=control

	#pragma HLS INTERFACE s_axilite port=stride_d bundle=control
	#pragma HLS INTERFACE s_axilite port=stride_h bundle=control
	#pragma HLS INTERFACE s_axilite port=stride_w bundle=control

	#pragma HLS INTERFACE s_axilite port=pad_d bundle=control
	#pragma HLS INTERFACE s_axilite port=pad_h bundle=control
	#pragma HLS INTERFACE s_axilite port=pad_w bundle=control

	#pragma HLS INTERFACE s_axilite port=return bundle=control

	// Compute output dimensions
	const int D_out = (D + 2 * pad_d - Kd) / stride_d + 1;
	const int H_out = (H + 2 * pad_h - Kh) / stride_h + 1;
	const int W_out = (W + 2 * pad_w - Kw) / stride_w + 1;

	// Simple buffers - NO partitioning
	data_type act_buf[MAX_IC][MAX_D][MAX_H][MAX_W];
	data_type weight_buf[MAX_OC][MAX_IC][MAX_KD][MAX_KH][MAX_KW];

	// Load activations - simple loop
	for (int ic = 0; ic < IC; ic++) {
		for (int d = 0; d < D; d++) {
			for (int h = 0; h < H; h++) {
				for (int w = 0; w < W; w++) {
					#pragma HLS PIPELINE II=1
					act_buf[ic][d][h][w] = activations[((ic*D + d)*H + h)*W + w];
				}
			}
		}
	}

	// Load weights - simple loop
	for (int oc = 0; oc < OC; oc++) {
		for (int ic = 0; ic < IC; ic++) {
			for (int kd = 0; kd < Kd; kd++) {
				for (int kh = 0; kh < Kh; kh++) {
					for (int kw = 0; kw < Kw; kw++) {
						#pragma HLS PIPELINE II=1
						weight_buf[oc][ic][kd][kh][kw] = weights[(((oc*IC + ic)*Kd + kd)*Kh + kh)*Kw + kw];
					}
				}
			}
		}
	}

	// Convolution - simple nested loops
	for (int oc = 0; oc < OC; oc++) {
		for (int od = 0; od < D_out; od++) {
			for (int oh = 0; oh < H_out; oh++) {
				for (int ow = 0; ow < W_out; ow++) {
					data_type sum = 0;
					
					for (int ic = 0; ic < IC; ic++) {
						for (int kd = 0; kd < Kd; kd++) {
							for (int kh = 0; kh < Kh; kh++) {
								for (int kw = 0; kw < Kw; kw++) {
									#pragma HLS PIPELINE II=1
									int id = od * stride_d + kd - pad_d;
									int ih = oh * stride_h + kh - pad_h;
									int iw = ow * stride_w + kw - pad_w;

									if (id >= 0 && id < D && ih >= 0 && ih < H && iw >= 0 && iw < W) {
										sum += act_buf[ic][id][ih][iw] * weight_buf[oc][ic][kd][kh][kw];
									}
								}
							}
						}
					}
					
					output_ws[((oc*D_out + od)*H_out + oh)*W_out + ow] = sum;
				}
			}
		}
	}
}



