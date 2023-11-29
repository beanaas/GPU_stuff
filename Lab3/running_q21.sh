#!/bin/bash

rm "output_q21.txt"
make clean; make bin/average
for((i=problem_size;i<=100000000;i*=2));do
    bin/average $i CPU  >> output_q21.txt
done
for((i=problem_size;i<=100000000;i*=2));do
    #echo "thread$i">>output.txt
    bin/average $i OPENMP  >> output_q21.txt
done
for((i=problem_size;i<=100000000;i*=2));do
    bin/average $i OPENCL  >> output_q21.txt
done

for((i=problem_size;i<=100000000;i*=2));do
    bin/average $i CUDA  >> output_q21.txt
done