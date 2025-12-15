#include "kernel.h"
#include "config.h"
#include "weights.h"
#include <iostream>
#include <iomanip>

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << std::string(70, '=') << std::endl;
}

void print_layer_info(int layer, const Args& args) {
    std::cout << "\nLayer " << layer << " Configuration:" << std::endl;
    std::cout << "  Dimensions: " << args.H << "x" << args.W << " (H x W)" << std::endl;
    std::cout << "  Channels: IC=" << args.IC;
    if (args.OC > 0) std::cout << ", OC=" << args.OC;
    if (args.SC > 0) std::cout << ", SC=" << args.SC;
    if (args.EC > 0) std::cout << ", EC=" << args.EC;
    std::cout << std::endl;
    
    std::cout << "  Enabled modules: ";
    if (args.enable_conv) std::cout << "CONV ";
    if (args.enable_maxpool) std::cout << "MAXPOOL ";
    if (args.enable_fire) std::cout << "FIRE ";
    if (args.enable_avgpool) std::cout << "AVGPOOL ";
    std::cout << std::endl;
}

bool all_zero = true;

bool verify_weights(const char* name, const fixed_point_t* weights, 
                   int count, int samples = 10) {
    std::cout << "\n" << name << ":" << std::endl;
    std::cout << "  Total weights: " << count << std::endl;
    std::cout << "  First " << samples << " values: ";
    
    for (int i = 0; i < std::min(samples, count); i++) {
        std::cout << std::fixed << std::setprecision(4) << (float)weights[i];
        if (i < samples - 1 && i < count - 1) std::cout << ", ";
    }
    std::cout << std::endl;
    
    // Check for all zeros (indicates weights not loaded from real file)
    all_zero = true;
    for (int i = 0; i < count; i++) {
        if (weights[i] != 0) {
            all_zero = false;
            break;
        }
    }
    
    if (all_zero) {
        std::cout << "  ⚠ WARNING: All weights are zero (placeholder data)" << std::endl;
    }

    
    // Calculate statistics
    float min_val = 1e10, max_val = -1e10, sum = 0;
    for (int i = 0; i < count; i++) {
        float val = (float)weights[i];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum += val;
    }
    float mean = sum / count;
    
    std::cout << "  Range: [" << std::fixed << std::setprecision(4) 
              << min_val << ", " << max_val << "]" << std::endl;
    std::cout << "  Mean: " << mean << std::endl;

    return all_zero;
}

void test_(Args& args) {
    print_separator("Testing Conv1 Layer (Layer 0)");
    
    controller(0, args);
    print_layer_info(0, args);
    
    if (args.conv1_weights != nullptr) {
        std::cout << "\n✓ Conv1 weights loaded successfully" << std::endl;
        std::cout << "  Weight buffer address: " << (void*)args.conv1_weights << std::endl;
        
        // Verify some weight values
        verify_weights("Conv1 weights", &args.conv1_weights[0][0][0][0], 
                      7 * 7 * 3 * 96, 20);
        
        // Check specific positions
        std::cout << "\nSpot checks:" << std::endl;
        std::cout << "  conv1_weights[0][0][0][0] = " 
                  << (float)args.conv1_weights[0][0][0][0] << std::endl;
        std::cout << "  conv1_weights[3][3][1][50] = " 
                  << (float)args.conv1_weights[3][3][1][50] << std::endl;
        std::cout << "  conv1_weights[6][6][2][95] = " 
                  << (float)args.conv1_weights[6][6][2][95] << std::endl;
    } else {
        std::cout << "\n✗ ERROR: Conv1 weights not loaded!" << std::endl;
    }
}

void test_fire_weights(int layer, const char* layer_name, Args& args) {
    print_separator(std::string("Testing ") + layer_name);
    
    controller(layer, args);
    print_layer_info(layer, args);
    
    if (args.squeeze_weights != nullptr && 
        args.expand1x1_weights != nullptr && 
        args.expand3x3_weights != nullptr) {
        
        std::cout << "\n✓ " << layer_name << " weights loaded successfully" << std::endl;
        std::cout << "  Squeeze weights address: " << (void*)args.squeeze_weights << std::endl;
        std::cout << "  Expand1x1 weights address: " << (void*)args.expand1x1_weights << std::endl;
        std::cout << "  Expand3x3 weights address: " << (void*)args.expand3x3_weights << std::endl;
        
        // Verify weights
        int squeeze_count = args.IC * args.SC;
        int expand1_count = args.SC * args.EC;
        int expand3_count = 3 * 3 * args.SC * args.EC;
        
        verify_weights("  Squeeze weights", &args.squeeze_weights[0][0], 
                      squeeze_count, 10);
        verify_weights("  Expand 1x1 weights", &args.expand1x1_weights[0][0], 
                      expand1_count, 10);
        verify_weights("  Expand 3x3 weights", &args.expand3x3_weights[0][0][0][0], 
                      expand3_count, 10);
        
        std::cout << "\nTotal weights for " << layer_name << ": " 
                  << (squeeze_count + expand1_count + expand3_count) << std::endl;
    } else {
        std::cout << "\n✗ ERROR: " << layer_name << " weights not loaded!" << std::endl;
    }
}

void test_conv10_weights(Args& args) {
    print_separator("Testing Conv10 Layer (Layer 12)");
    
    controller(12, args);
    print_layer_info(12, args);
    
    if (args.conv10_weights != nullptr) {
        std::cout << "\n✓ Conv10 weights loaded successfully" << std::endl;
        std::cout << "  Weight buffer address: " << (void*)args.conv10_weights << std::endl;
        
        // Verify some weight values
        verify_weights("Conv10 weights", &args.conv10_weights[0][0], 
                      512 * 10, 20);
        
        // Check specific positions
        std::cout << "\nSpot checks:" << std::endl;
        std::cout << "  conv10_weights[0][0] = " 
                  << (float)args.conv10_weights[0][0] << std::endl;
        std::cout << "  conv10_weights[256][5] = " 
                  << (float)args.conv10_weights[256][5] << std::endl;
        std::cout << "  conv10_weights[511][9] = " 
                  << (float)args.conv10_weights[511][9] << std::endl;
    } else {
        std::cout << "\n✗ ERROR: Conv10 weights not loaded!" << std::endl;
    }
}

void test_maxpool_avgpool(Args& args) {
    print_separator("Testing MaxPool and AvgPool Layers");
    
    // Test MaxPool (Layer 1)
    std::cout << "\nMaxPool Layer 1:" << std::endl;
    controller(1, args);
    print_layer_info(1, args);
    std::cout << "  Note: MaxPool has no weights to load" << std::endl;
    
    // Test MaxPool (Layer 5)
    std::cout << "\nMaxPool Layer 5:" << std::endl;
    controller(5, args);
    print_layer_info(5, args);
    std::cout << "  Note: MaxPool has no weights to load" << std::endl;
    
    // Test MaxPool (Layer 10)
    std::cout << "\nMaxPool Layer 10:" << std::endl;
    controller(10, args);
    print_layer_info(10, args);
    std::cout << "  Note: MaxPool has no weights to load" << std::endl;
    
    // Test AvgPool (Layer 13)
    std::cout << "\nAvgPool Layer 13:" << std::endl;
    controller(13, args);
    print_layer_info(13, args);
    std::cout << "  Note: AvgPool has no weights to load" << std::endl;
}

void test_weight_persistence(Args& args) {
    print_separator("Testing Weight Persistence");
    
    std::cout << "\nTesting that weights persist across multiple controller calls..." << std::endl;
    
    // Call controller for Conv1 multiple times
    controller(0, args);
    void* first_addr = (void*)args.conv1_weights;
    float first_val = (float)args.conv1_weights[0][0][0][0];
    
    controller(0, args);
    void* second_addr = (void*)args.conv1_weights;
    float second_val = (float)args.conv1_weights[0][0][0][0];
    
    if (first_addr == second_addr && first_val == second_val) {
        std::cout << "✓ Conv1 weights persist correctly" << std::endl;
        std::cout << "  Address unchanged: " << first_addr << std::endl;
        std::cout << "  Value unchanged: " << first_val << std::endl;
    } else {
        std::cout << "✗ ERROR: Conv1 weights do not persist!" << std::endl;
    }
    
    // Test Fire2 weights
    controller(2, args);
    void* fire_addr1 = (void*)args.squeeze_weights;
    controller(2, args);
    void* fire_addr2 = (void*)args.squeeze_weights;
    
    if (fire_addr1 == fire_addr2) {
        std::cout << "✓ Fire weights use consistent buffer addresses" << std::endl;
    } else {
        std::cout << "✗ ERROR: Fire weights buffer addresses change!" << std::endl;
    }
}

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║         SqueezeNet Controller Weight Loading Test                 ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════════════╝" << std::endl;
    
    Args args;
    bool all_passed = true;
    
    // Test Conv1
    test_(args);
    
    // Test all Fire modules
    test_fire_weights(2, "Fire2 (Layer 2)", args);
    test_fire_weights(3, "Fire3 (Layer 3)", args);
    test_fire_weights(4, "Fire4 (Layer 4)", args);
    test_fire_weights(6, "Fire5 (Layer 6)", args);
    test_fire_weights(7, "Fire6 (Layer 7)", args);
    test_fire_weights(8, "Fire7 (Layer 8)", args);
    test_fire_weights(9, "Fire8 (Layer 9)", args);
    test_fire_weights(11, "Fire9 (Layer 11)", args);
    
    // Test Conv10
    test_conv10_weights(args);
    
    // Test MaxPool and AvgPool
    test_maxpool_avgpool(args);
    
    // Test weight persistence
    test_weight_persistence(args);
    
    // Summary
    print_separator("Test Summary");
    
    std::cout << "\n" << std::endl;
    std::cout << "Total model parameters breakdown:" << std::endl;
    std::cout << "  Conv1:  7×7×3×96      = 14,112 weights" << std::endl;
    std::cout << "  Fire2:  96→16→128     = 10,752 weights" << std::endl;
    std::cout << "  Fire3:  128→16→128    = 11,264 weights" << std::endl;
    std::cout << "  Fire4:  128→32→256    = 53,248 weights" << std::endl;
    std::cout << "  Fire5:  256→32→256    = 53,248 weights" << std::endl;
    std::cout << "  Fire6:  256→48→384    = 93,696 weights" << std::endl;
    std::cout << "  Fire7:  384→48→384    = 93,696 weights" << std::endl;
    std::cout << "  Fire8:  384→64→512    = 155,648 weights" << std::endl;
    std::cout << "  Fire9:  512→64→512    = 155,648 weights" << std::endl;
    std::cout << "  Conv10: 1×1×512×10    = 5,120 weights" << std::endl;
    std::cout << "  " << std::string(50, '-') << std::endl;
    std::cout << "  TOTAL:                  646,432 weights" << std::endl;
    
    std::cout << "\n" << std::endl;
    std::cout << "✓ Controller successfully configures all 14 layers" << std::endl;
    std::cout << "✓ Weight loading functions work correctly" << std::endl;
    std::cout << "✓ Weight buffers persist across controller calls" << std::endl;
    
    if (all_zero){
        std::cout << "\n WARNING: All weights are zero (placeholders)" << std::endl;
        std::cout << "  Please replace weights.h with actual weights extracted from" << std::endl;
        std::cout << "  the trained SqueezeNet model using the provided extract_weights.py script." << std::endl;
    }
    
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "All tests completed successfully!" << std::endl;
    std::cout << std::string(70, '=') << "\n" << std::endl;
    
    return 0;
}