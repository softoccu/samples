# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -fopenmp

# Source files
SRCS = $(wildcard *.cpp)

# Executable files
BIN_DIR = ./bin
TARGETS = $(SRCS:%.cpp=$(BIN_DIR)/%)

# Default target
all: $(TARGETS)

# Create bin directory if it doesn't exist
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Rule to build each executable
$(BIN_DIR)/%: %.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

# Clean target
clean:
	rm -rf $(BIN_DIR)

.PHONY: all clean