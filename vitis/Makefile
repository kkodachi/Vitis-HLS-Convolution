# Compiler
CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wno-unused-variable -Wno-unused-label -Wno-unknown-pragmas

# Target executable
TARGET := testbench

# Source files
SRCS := testbench.cpp conv3d.cpp fire.cpp maxpool.cpp avgpool.cpp

# Object files
OBJS := $(SRCS:.cpp=.o)

# Default rule
all: $(TARGET)

# Link object files into executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile .cpp files into .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJS) $(TARGET)

# Phony targets
.PHONY: all clean
