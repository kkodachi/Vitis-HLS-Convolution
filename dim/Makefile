# Compiler
CXX := clang++
CXXFLAGS := -std=c++17 -O2 -Wall -Wno-unused-variable -Wno-unused-label -Wno-unknown-pragmas -stdlib=libc++

# Target executables
TARGET := tb_controller
TARGET_OLD := testbench
TARGET_FULL := tb_integrate
TARGET_LITE := tb_lite
TARGET_WEIGHTS := tb_extract

# Source files
SRCS := conv3d.cpp fire.cpp maxpool.cpp avgpool.cpp controller.cpp weights.cpp
SRCS_OLD := testbench.cpp conv3d.cpp fire.cpp maxpool.cpp avgpool.cpp
SRCS_WEIGHTS := controller.cpp

# Testbench source
TB_SRC := tb_controller.cpp
TB_FULL_SRC := tb_integrate.cpp
TB_LITE_SRC := tb_lite.cpp
TB_WEIGHTS_SRC := tb_extract.cpp

# Object files
OBJS := $(SRCS:.cpp=.o)
OBJS_OLD := $(SRCS_OLD:.cpp=.o)
OBJS_WEIGHTS := $(SRCS_WEIGHTS:.cpp=.o)
TB_OBJ := $(TB_SRC:.cpp=.o)
TB_FULL_OBJ := $(TB_FULL_SRC:.cpp=.o)
TB_LITE_OBJ := $(TB_LITE_SRC:.cpp=.o)
TB_WEIGHTS_OBJ := $(TB_WEIGHTS_SRC:.cpp=.o)

# Default rule
all: $(TARGET)

# Also provide option to build old testbench
old: $(TARGET_OLD)

# Build full system testbench
full: $(TARGET_FULL)

# Build lite testbench (memory efficient)
lite: $(TARGET_LITE)

# Build weights loading testbench
weights: $(TARGET_WEIGHTS)

# Link testbench_controller with module objects
$(TARGET): $(TB_OBJ) $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Link old testbench
$(TARGET_OLD): $(OBJS_OLD)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Link full system testbench
$(TARGET_FULL): $(TB_FULL_OBJ) $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Link lite testbench
$(TARGET_LITE): $(TB_LITE_OBJ) $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Link weights loading testbench
$(TARGET_WEIGHTS): $(TB_WEIGHTS_OBJ) $(OBJS_WEIGHTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile .cpp files into .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJS) $(OBJS_OLD) $(OBJS_WEIGHTS) $(TB_OBJ) $(TB_FULL_OBJ) $(TB_LITE_OBJ) $(TB_WEIGHTS_OBJ) $(TB_WEIGHT_LOAD_OBJ) $(TARGET) $(TARGET_OLD) $(TARGET_FULL) $(TARGET_LITE) $(TARGET_WEIGHTS)

# Run the controller testbench
run: $(TARGET)
	./$(TARGET)

# Run the old testbench
run_old: $(TARGET_OLD)
	./$(TARGET_OLD)

# Run the full system testbench
run_full: $(TARGET_FULL)
	./$(TARGET_FULL)

# Run the lite testbench (recommended for MacBook)
run_lite: $(TARGET_LITE)
	./$(TARGET_LITE)

# Run the weights loading testbench
run_weights: $(TARGET_WEIGHTS)
	./$(TARGET_WEIGHTS)

# Phony targets
.PHONY: all old full lite weights weight_load clean run run_old run_full run_lite run_weights