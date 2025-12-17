#include "config.h"
#include "kernel.h"
#include <cstdlib>
#include <iostream>

void init_array(fixed_point_t* arr, int elements){
	for (int i = 0; i < elements; i++){
		arr[i] = std::rand() % 256 - 128;
	}
}

int main()
{
	std::srand(0);

	fixed_point_t input[CONV1_DIM * CONV1_DIM * CONV1_IC];
	fixed_point_t output[NUM_CLASSES];
	fixed_point_t conv1w[MAX_CONV_K * MAX_CONV_K * CONV1_IC * CONV1_OC];
	fixed_point_t conv2w[MAX_CONV_K * MAX_CONV_K * CONV2_IC *CONV2_OC];
	fixed_point_t conv3w[MAX_CONV_K * MAX_CONV_K * CONV3_IC *CONV3_OC];
	fixed_point_t conv4w[MAX_CONV_K * MAX_CONV_K * CONV4_IC *CONV4_OC];
	fixed_point_t conv5w[CONV1_IC * NUM_CLASSES];

	init_array(input,CONV1_DIM * CONV1_DIM * CONV1_IC);
	init_array(conv1w,MAX_CONV_K * MAX_CONV_K * CONV1_IC * CONV1_OC);
	init_array(conv2w,MAX_CONV_K * MAX_CONV_K * CONV2_IC *CONV2_OC);
	init_array(conv3w,MAX_CONV_K * MAX_CONV_K * CONV3_IC *CONV3_OC);
	init_array(conv4w,MAX_CONV_K * MAX_CONV_K * CONV4_IC *CONV4_OC);
	init_array(conv5w,CONV1_IC * NUM_CLASSES);

	for (int i = 0; i < NUM_CLASSES; i++){
		output[i] = 0;
	}

	squeezenet(
	    input,
	    output,
	    conv1w,
	    conv2w,
	    conv3w,
	    conv4w,
	    conv5w
	);

	for (int i = 0; i < NUM_CLASSES; i++){
		std::cout << "output[" << i << "]: " << int(output[i]) << std::endl;
	}
}
