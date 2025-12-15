#ifndef AP_FIXED_MOCK_H
#define AP_FIXED_MOCK_H

// Mock ap_fixed for testing on systems without Xilinx HLS
// This allows compilation on MacBook/Linux for functional testing

#include <cmath>

// Mock rounding and saturation modes
enum ap_q_mode { AP_RND };
enum ap_o_mode { AP_SAT };

// Simple mock of ap_fixed using float
template<int W, int I, ap_q_mode Q = AP_RND, ap_o_mode O = AP_SAT>
class ap_fixed {
private:
    float value;
    
    // Calculate min/max based on integer bits
    static constexpr float get_max() {
        return std::pow(2.0f, I) - std::pow(2.0f, -(W-I));
    }
    
    static constexpr float get_min() {
        return -std::pow(2.0f, I);
    }
    
    // Saturate value to valid range
    float saturate(float v) const {
        float max_val = get_max();
        float min_val = get_min();
        if (v > max_val) return max_val;
        if (v < min_val) return min_val;
        return v;
    }

public:
    // Constructors
    ap_fixed() : value(0.0f) {}
    ap_fixed(float v) : value(saturate(v)) {}
    ap_fixed(double v) : value(saturate((float)v)) {}
    ap_fixed(int v) : value(saturate((float)v)) {}
    ap_fixed(long v) : value(saturate((float)v)) {}
    ap_fixed(unsigned int v) : value(saturate((float)v)) {}
    
    // Copy constructor
    ap_fixed(const ap_fixed& other) : value(other.value) {}
    
    // Assignment operators
    ap_fixed& operator=(float v) {
        value = saturate(v);
        return *this;
    }
    
    ap_fixed& operator=(double v) {
        value = saturate((float)v);
        return *this;
    }
    
    ap_fixed& operator=(int v) {
        value = saturate((float)v);
        return *this;
    }
    
    ap_fixed& operator=(const ap_fixed& other) {
        value = other.value;
        return *this;
    }
    
    // Conversion operators
    operator float() const { return value; }
    operator double() const { return (double)value; }
    operator int() const { return (int)value; }
    
    // Arithmetic operators
    ap_fixed operator+(const ap_fixed& other) const {
        return ap_fixed(value + other.value);
    }
    
    ap_fixed operator-(const ap_fixed& other) const {
        return ap_fixed(value - other.value);
    }
    
    ap_fixed operator*(const ap_fixed& other) const {
        return ap_fixed(value * other.value);
    }
    
    ap_fixed operator/(const ap_fixed& other) const {
        return ap_fixed(value / other.value);
    }
    
    // Compound assignment operators
    ap_fixed& operator+=(const ap_fixed& other) {
        value = saturate(value + other.value);
        return *this;
    }
    
    ap_fixed& operator-=(const ap_fixed& other) {
        value = saturate(value - other.value);
        return *this;
    }
    
    ap_fixed& operator*=(const ap_fixed& other) {
        value = saturate(value * other.value);
        return *this;
    }
    
    ap_fixed& operator/=(const ap_fixed& other) {
        value = saturate(value / other.value);
        return *this;
    }
    
    // Comparison operators
    bool operator==(const ap_fixed& other) const {
        return value == other.value;
    }
    
    bool operator!=(const ap_fixed& other) const {
        return value != other.value;
    }
    
    bool operator<(const ap_fixed& other) const {
        return value < other.value;
    }
    
    bool operator>(const ap_fixed& other) const {
        return value > other.value;
    }
    
    bool operator<=(const ap_fixed& other) const {
        return value <= other.value;
    }
    
    bool operator>=(const ap_fixed& other) const {
        return value >= other.value;
    }
    
    // Unary operators
    ap_fixed operator-() const {
        return ap_fixed(-value);
    }
};

#endif // AP_FIXED_MOCK_H