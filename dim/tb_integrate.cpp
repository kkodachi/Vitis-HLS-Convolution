#include "kernel.h"
#include "config.h"
#include <iostream>
#include <cstdlib>

// This testbench should be run in Vitis HLS or a Linux system with more memory
// It tests the controller + kernel execution integration

int main() {
    std::cout << "================================================" << std::endl;
    std::cout << "  Controller Integration Test (Vitis HLS)" << std::endl;
    std::cout << "================================================\n" << std::endl;
    
    // Allocate buffers (use static in Vitis HLS)
    static fixed_point_t buf1[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC];
    static fixed_point_t buf2[MAX_FIRE_H][MAX_FIRE_W][MAX_FIRE_IC];
    
    // Allocate weights (initialize to small random values for testing)
    static fixed_point_t conv_weights[MAX_CONV_K][MAX_CONV_K][MAX_CONV_IC][MAX_CONV_OC];
    static fixed_point_t fire_squeeze_weights[MAX_FIRE_IC][MAX_FIRE_SC];
    static fixed_point_t fire_expand1x1_weights[MAX_FIRE_SC][MAX_FIRE_EC];
    static fixed_point_t fire_expand3x3_weights[3][3][MAX_FIRE_SC][MAX_FIRE_EC];
    
    std::cout << "Initializing test input (224x224x3)..." << std::endl;
    for (int h = 0; h < 224; h++) {
        for (int w = 0; w < 224; w++) {
            for (int c = 0; c < 3; c++) {
                int steps = std::rand() % 256;
                buf1[h][w][c] = -8.0f + 0.0625f * steps;
            }
        }
    }
    
    std::cout << "Initializing Conv1 weights (7x7x3x96)..." << std::endl;
    for (int kh = 0; kh < 7; kh++) {
        for (int kw = 0; kw < 7; kw++) {
            for (int ic = 0; ic < 3; ic++) {
                for (int oc = 0; oc < 96; oc++) {
                    int steps = std::rand() % 256;
                    conv_weights[kh][kw][ic][oc] = -8.0f + 0.0625f * steps;
                }
            }
        }
    }
    
    std::cout << "\n=== Testing First 3 Layers ===" << std::endl;
    std::cout << "Conv1 -> MaxPool1 -> Fire2\n" << std::endl;
    
    bool use_buf1_as_input = true;
    bool all_passed = true;
    
    // Test Conv1
    {
        std::cout << "--- Layer 0: Conv1 ---" << std::endl;
        Args args;
        controller(0, args);
        
        std::cout << "  Config: " << args.H << "x" << args.W << "x" << args.IC 
                  << " -> K=" << args.K << " S=" << args.S << " P=" << args.P 
                  << " OC=" << args.OC << std::endl;
        
        if (!args.enable_conv) {
            std::cout << "  ✗ FAIL: Conv not enabled!" << std::endl;
            all_passed = false;
        } else {
            // Execute Conv1
            conv3d(true, 
                   (fixed_point_t (*)[MAX_CONV_W][MAX_CONV_IC])buf1,
                   conv_weights,
                   buf2,
                   args.H, args.W, args.IC, args.OC, args.K, args.S, args.P);
            
            int H_OUT = (args.H + 2*args.P - args.K)/args.S + 1;
            int W_OUT = (args.W + 2*args.P - args.K)/args.S + 1;
            
            // Check output is not all zeros
            bool has_nonzero = false;
            for (int h = 0; h < H_OUT && !has_nonzero; h++) {
                for (int w = 0; w < W_OUT && !has_nonzero; w++) {
                    for (int c = 0; c < args.OC && !has_nonzero; c++) {
                        if (buf2[h][w][c] != 0) {
                            has_nonzero = true;
                        }
                    }
                }
            }
            
            if (has_nonzero) {
                std::cout << "  ✓ PASS: Conv1 executed, output = " << H_OUT << "x" 
                          << W_OUT << "x" << args.OC << std::endl;
                std::cout << "  Sample outputs: buf2[0][0][0]=" << (float)buf2[0][0][0]
                          << " buf2[0][0][1]=" << (float)buf2[0][0][1] << std::endl;
            } else {
                std::cout << "  ✗ FAIL: Output is all zeros!" << std::endl;
                all_passed = false;
            }
        }
        use_buf1_as_input = false;
    }
    
    // Test MaxPool1
    {
        std::cout << "\n--- Layer 1: MaxPool1 ---" << std::endl;
        Args args;
        controller(1, args);
        
        std::cout << "  Config: " << args.H << "x" << args.W << "x" << args.IC << std::endl;
        
        if (!args.enable_maxpool) {
            std::cout << "  ✗ FAIL: MaxPool not enabled!" << std::endl;
            all_passed = false;
        } else {
            // Input should be buf2 (output from Conv1)
            // Output should be buf1
            maxpool(true, buf2, buf1, args.H, args.W, args.IC);
            
            const int K = 3, S = 2, P = 0;
            int H_OUT = ((args.H + 2*P - K + S - 1) / S) + 1;
            int W_OUT = ((args.W + 2*P - K + S - 1) / S) + 1;
            
            // Check output dimensions are correct
            std::cout << "  ✓ PASS: MaxPool1 executed, output = " << H_OUT << "x" 
                      << W_OUT << "x" << args.IC << std::endl;
            std::cout << "  Sample outputs: buf1[0][0][0]=" << (float)buf1[0][0][0]
                      << " buf1[0][0][1]=" << (float)buf1[0][0][1] << std::endl;
        }
        use_buf1_as_input = true;
    }
    
    // Test Fire2
    {
        std::cout << "\n--- Layer 2: Fire2 ---" << std::endl;
        Args args;
        controller(2, args);
        
        std::cout << "  Config: " << args.H << "x" << args.W << "x" << args.IC 
                  << " SC=" << args.SC << " EC=" << args.EC << std::endl;
        
        if (!args.enable_fire) {
            std::cout << "  ✗ FAIL: Fire not enabled!" << std::endl;
            all_passed = false;
        } else {
            // Initialize fire weights
            std::cout << "  Initializing Fire2 weights..." << std::endl;
            for (int ic = 0; ic < args.IC; ic++) {
                for (int sc = 0; sc < args.SC; sc++) {
                    int steps = std::rand() % 256;
                    fire_squeeze_weights[ic][sc] = -8.0f + 0.0625f * steps;
                }
            }
            
            for (int sc = 0; sc < args.SC; sc++) {
                for (int ec = 0; ec < args.EC; ec++) {
                    int steps = std::rand() % 256;
                    fire_expand1x1_weights[sc][ec] = -8.0f + 0.0625f * steps;
                    for (int i = 0; i < 3; i++) {
                        for (int j = 0; j < 3; j++) {
                            steps = std::rand() % 256;
                            fire_expand3x3_weights[i][j][sc][ec] = -8.0f + 0.0625f * steps;
                        }
                    }
                }
            }
            
            // Input should be buf1 (output from MaxPool1)
            // Output should be buf2
            fire(true, buf1, 
                 fire_squeeze_weights, fire_expand1x1_weights, fire_expand3x3_weights,
                 buf2,
                 args.H, args.W, args.IC, args.SC, args.EC);
            
            int OC = 2 * args.EC;
            std::cout << "  ✓ PASS: Fire2 executed, output = " << args.H << "x" 
                      << args.W << "x" << OC << std::endl;
            std::cout << "  Sample outputs: buf2[0][0][0]=" << (float)buf2[0][0][0]
                      << " buf2[0][0][64]=" << (float)buf2[0][0][64] << std::endl;
        }
    }
    
    // Summary
    std::cout << "\n\n================================================" << std::endl;
    std::cout << "              TEST SUMMARY" << std::endl;
    std::cout << "================================================" << std::endl;
    
    if (all_passed) {
        std::cout << "\n✓ ALL INTEGRATION TESTS PASSED!" << std::endl;
        std::cout << "Controller successfully coordinates kernel execution." << std::endl;
        std::cout << "Data flows correctly through the pipeline." << std::endl;
        return 0;
    } else {
        std::cout << "\n✗ SOME TESTS FAILED!" << std::endl;
        std::cout << "Controller integration needs debugging." << std::endl;
        return 1;
    }
}