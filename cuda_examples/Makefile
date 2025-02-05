# Compiler
NVCC = nvcc

# Source and binary directories
SRC_DIR = .
BIN_DIR = ./bin

# Find all .cu files in the source directory
SOURCES = $(wildcard $(SRC_DIR)/*.cu)

# Define the targets (executables) by replacing .cu with empty string in the filenames
TARGETS = $(patsubst $(SRC_DIR)/%.cu, $(BIN_DIR)/%, $(SOURCES))

# Default target
all: $(TARGETS)

# Rule to build the binary directory if it doesn't exist
$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

# Rule to compile cudaFFT.cu with linking to FFT library
$(BIN_DIR)/cudaFFT: $(SRC_DIR)/cudaFFT.cu | $(BIN_DIR)
	$(NVCC) $< -o $@ -lcufft

# Rule to compile other .cu files into executables
$(BIN_DIR)/%: $(SRC_DIR)/%.cu | $(BIN_DIR)
	$(NVCC) $< -o $@

# Ensure cudaFFT.cu is handled separately
$(BIN_DIR)/%: $(SRC_DIR)/cudaFFT.cu
	@true

# Phony target to clean the build
clean:
	@rm -rf $(BIN_DIR)/*

# Phony targets
.PHONY: all clean