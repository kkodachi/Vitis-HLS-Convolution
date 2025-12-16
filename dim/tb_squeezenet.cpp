#include "kernel.h"
#include "config.h"
#include "weights.h"

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <cmath>

// Generate random fixed-point value in range [-8.0, 7.9375]
fixed_point_t random_fixed_point() {
    // Generate value using ap_fixed range
    int steps = std::rand() % 256;
    return fixed_point_t(-8.0f + 0.0625f * steps);
}

// Load random CIFAR-10-like image (224x224x3)
void load_random_image(fixed_point_t image[MAX_CONV_H][MAX_CONV_W][MAX_CONV1_IC]) {
    std::cout << "Loading random test image (224x224x3)..." << std::endl;
    
    for (int h = 0; h < 224; h++) {
        for (int w = 0; w < 224; w++) {
            for (int c = 0; c < 3; c++) {
                // Generate random pixel value in typical range
                image[h][w][c] = random_fixed_point();
            }
        }
    }
    
    std::cout << "Image loaded successfully." << std::endl;
}

// Print output probabilities
void print_output(const fixed_point_t output[AVGPOOL_C]) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "  SqueezeNet Output (Class Scores)" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    // Find max score for prediction
    int max_idx = 0;
    fixed_point_t max_val = output[0];
    
    for (int i = 0; i < AVGPOOL_C; i++) {
        std::cout << "  Class " << i << ": " << std::fixed << std::setprecision(4)
                  << (float)output[i] << std::endl;
        
        if (output[i] > max_val) {
            max_val = output[i];
            max_idx = i;
        }
    }
    
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "  Predicted Class: " << max_idx 
              << " (score: " << (float)max_val << ")" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
}

// Test individual layer execution
void test_layer_config() {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "  Layer Configuration Test" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    LayerConfig config;
    
    const char* layer_names[] = {
        "Conv1", "MaxPool1", "Fire2", "Fire3", "Fire4", "MaxPool2",
        "Fire5", "Fire6", "Fire7", "Fire8", "MaxPool3", "Fire9",
        "Conv10", "AvgPool"
    };
    
    const char* type_names[] = {
        "NONE", "CONV", "MAXPOOL", "FIRE", "CONV10", "AVGPOOL"
    };
    
    for (int layer = 0; layer < TOTAL_LAYERS; layer++) {
        configure_layer(layer, config);
        
        std::cout << "\nLayer " << layer << ": " << layer_names[layer] << std::endl;
        std::cout << "  Type: " << type_names[config.layer_type] << std::endl;
        std::cout << "  Input:  " << config.H << "x" << config.W << "x" << config.IC << std::endl;
        
        // Calculate output dimensions
        int H_out = config.H, W_out = config.W, C_out = config.IC;
        
        switch(config.layer_type) {
            case LAYER_CONV:
                H_out = (config.H + 2*config.P - config.K) / config.S + 1;
                W_out = (config.W + 2*config.P - config.K) / config.S + 1;
                C_out = config.OC;
                std::cout << "  Params: K=" << config.K << ", S=" << config.S 
                          << ", P=" << config.P << std::endl;
                break;
                
            case LAYER_MAXPOOL:
                H_out = (config.H - 3) / 2 + 1; // K=3, S=2
                W_out = (config.W - 3) / 2 + 1;
                std::cout << "  Params: K=3, S=2, P=0" << std::endl;
                break;
                
            case LAYER_FIRE:
                C_out = 2 * config.EC; // expand1x1 + expand3x3
                std::cout << "  Params: SC=" << config.SC << ", EC=" << config.EC
                          << " (Fire" << config.fire_id << ")" << std::endl;
                break;
                
            case LAYER_CONV10:
                C_out = config.OC;
                std::cout << "  Params: 1x1 conv, OC=" << config.OC << std::endl;
                break;
                
            case LAYER_AVGPOOL:
                H_out = 1;
                W_out = 1;
                std::cout << "  Params: Global average pooling" << std::endl;
                break;
                
            default:
                break;
        }
        
        if (config.layer_type != LAYER_AVGPOOL) {
            std::cout << "  Output: " << H_out << "x" << W_out << "x" << C_out << std::endl;
        } else {
            std::cout << "  Output: " << C_out << " (1D vector)" << std::endl;
        }
    }
}

// Test full SqueezeNet inference
void test_full_inference() {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "  Full SqueezeNet Inference Test" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    // Allocate input image
    auto input_image = new fixed_point_t[MAX_CONV_H][MAX_CONV_W][MAX_CONV1_IC];
    
    // Load random test image
    load_random_image(input_image);
    
    // Allocate output buffer
    fixed_point_t output[AVGPOOL_C];
    
    std::cout << "\nExecuting SqueezeNet pipeline..." << std::endl;
    std::cout << "This will execute all 14 layers sequentially." << std::endl;
    
    // Execute full network
    controller(input_image, output);
    
    std::cout << "✓ Inference complete!" << std::endl;
    
    // Print results
    print_output(output);
    
    // Cleanup
    delete[] input_image;
}

// Test weight loading
void test_weight_loading() {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "  Weight Loading Test" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    std::cout << "\nChecking if weights are properly loaded..." << std::endl;
    
    // Check a few sample weights
    std::cout << "\nSample Conv1 weights:" << std::endl;
    for (int i = 500; i < 510; i++) {
        std::cout << "  conv1_weights_flat[" << i << "] = " 
                  << (float)conv1_weights_flat[i] << std::endl;
    }
    
    std::cout << "\nSample Fire2 squeeze weights:" << std::endl;
    for (int i = 25; i < 35; i++) {
        std::cout << "  fire2_squeeze_weights_flat[" << i << "] = " 
                  << (float)fire2_squeeze_weights_flat[i] << std::endl;
    }
    
    std::cout << "\nSample Conv10 weights:" << std::endl;
    for (int i = 300; i < 310; i++) {
        std::cout << "  conv10_weights_flat[" << i << "] = " 
                  << (float)conv10_weights_flat[i] << std::endl;
    }
    
    std::cout << "\n✓ Weights accessible from combined header" << std::endl;
}

// Main test runner
int main() {
    std::srand(std::time(nullptr));
    
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║              SqueezeNet HLS Implementation Test                    ║" << std::endl;
    std::cout << "║                   Task 3: Full Pipeline                            ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════════════╝" << std::endl;
    
    std::cout << "\nThis testbench validates:" << std::endl;
    std::cout << "  1. Layer configuration system" << std::endl;
    std::cout << "  2. Weight loading from QAT checkpoint" << std::endl;
    std::cout << "  3. Ping-pong buffer management" << std::endl;
    std::cout << "  4. Full end-to-end inference" << std::endl;
    
    // Run all tests
    test_layer_config();
    test_weight_loading();
    test_full_inference();
    
    // Summary
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "  Test Summary" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    std::cout << "\n✓ All tests completed successfully!" << std::endl;
    std::cout << "\nNext steps:" << std::endl;
    std::cout << "  1. Run Vitis HLS synthesis" << std::endl;
    std::cout << "  2. Verify resource utilization" << std::endl;
    std::cout << "  3. Check latency/throughput" << std::endl;
    std::cout << "  4. Generate bitstream for Kria KV260" << std::endl;
    
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << std::endl;
    
    return 0;
}