# Compiler
CXX = g++
CXXFLAGS = -std=c++11 -DVITIS_HLS \
           -I/home/ee511/Vitis/2021.1/include \
           -I/home/ee511/Vitis_HLS/2021.1/include \
           -I/home/ee511/Documents/project511/Vitis-HLS-Convolution/dim

# Source files
SRCS = /home/ee511/Documents/project511/Vitis-HLS-Convolution/dim/testbench.cpp \
       /home/ee511/Documents/project511/Vitis-HLS-Convolution/dim/conv3d.cpp \
       /home/ee511/Documents/project511/Vitis-HLS-Convolution/dim/avgpool.cpp \
       /home/ee511/Documents/project511/Vitis-HLS-Convolution/dim/maxpool.cpp \
       /home/ee511/Documents/project511/Vitis-HLS-Convolution/dim/fire.cpp \
       /home/ee511/Documents/project511/Vitis-HLS-Convolution/dim/conv1.cpp \
       /home/ee511/Documents/project511/Vitis-HLS-Convolution/dim/conv10.cpp

# Output executable
TARGET = tb

# Default target
all: $(TARGET)

# Build rule
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

# Clean rule
clean:
	rm -f $(TARGET) *.o

.PHONY: all clean

