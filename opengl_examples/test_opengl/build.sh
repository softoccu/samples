#!/bin/bash

# Set the source file and output binary
SOURCE_FILE="test_opengl.cpp"
OUTPUT_BINARY="testOpenGL"

# Compile the program
g++ -o $OUTPUT_BINARY $SOURCE_FILE -lGL -lGLEW -lglfw

# Check if the compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful."
else
    echo "Compilation failed."
fi
