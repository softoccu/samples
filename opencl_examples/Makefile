# Compiler
CXX := g++

# Compiler flags
CXXFLAGS := -std=c++17

# Linker flags
LDFLAGS := -lOpenCL

# Source files
SRCS := $(wildcard *.cpp)

# Object files
OBJS := $(SRCS:.cpp=.o)

# Executable files
BINS := $(patsubst %.cpp, ./bin/%, $(SRCS))

# Create bin directory if it doesn't exist
$(shell mkdir -p bin)

# Default target
all: $(BINS)

# Rule to build executables
./bin/%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# Clean up
clean:
	rm -f $(OBJS) $(BINS)