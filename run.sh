#!/bin/bash

output_data=()

# Loop for 8 iterations
for index in {0..16}
do
    echo "Iteration $index"

    # Run the commands and save the output to a file
    make clean
    rm mandelbrot.ppm
    make MEASURE=1 NB_THREADS=$index LOADBALANCE=2
    output_data+=("$(./mandelbrot-*)")

    echo "Iteration $index completed"
done

printf '%s\n' "${output_data[@]}" > output.txt