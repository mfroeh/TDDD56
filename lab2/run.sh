#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <MEASURE> <NON_BLOCKING> <NUM_PUSH_POP>"
    exit 1
fi

for index in {0..16}
do
    echo "Iteration $index"

    make clean 
    make NB_THREADS=$index MEASURE=$1 NON_BLOCKING=$2 MAX_PUSH_POP=$3

    output=$(./stack)
    max_time=$(echo "$output" | grep "Thread" | awk '{print $4}' | sort -n | tail -n 1)
    output_data+=("$max_time")

    echo "Iteration $index completed"
done

mkdir -p "measurements/"
filename="measurements/m_$1nb_$2_mpp_$3.txt"
printf "MEASURE: $1, NON_BLOCKING = $2, MAX_PUSH_POP = $3\n" > $filename
printf '%s\n' "${output_data[@]}" >> $filename

printf "$filename"
