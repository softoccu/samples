#!/bin/bash

# Set the source file and output binary
SOURCE_FILE="bouncing_ball.cpp"
OUTPUT_BINARY="BouncingBall"

# Compile the program
g++ -o $OUTPUT_BINARY $SOURCE_FILE -lGL -lGLEW -lglfw -lm

# Check if the compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful. Running the program..."
    ./$OUTPUT_BINARY
else
    echo "Compilation failed."
fi
