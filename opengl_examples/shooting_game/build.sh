#!/bin/bash

# Set the source file and output binary
SOURCE_FILE="shooting_game.cpp"
OUTPUT_BINARY="shooting_game"

# Compile the program
g++ -o $OUTPUT_BINARY $SOURCE_FILE -lGL -lGLEW -lglfw -lm
g++ -o shooting_game_smallwindow shooting_game_smallwindow.cpp -lGL -lGLEW -lglfw -lm
# Check if the compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful."
else
    echo "Compilation failed."
fi
