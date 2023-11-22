#!/bin/bash

problem_size=100000
rm "output.txt"
make clean; make bin/dotproduct
for((i=problem_size;i<=100000000;i*=2));do
    bin/dotproduct $i CPU  >> output.txt
done
for((i=problem_size;i<=100000000;i*=2));do
    #echo "thread$i">>output.txt
    bin/dotproduct $i OPENMP  >> output.txt
done
for((i=problem_size;i<=100000000;i*=2));do
    bin/dotproduct $i OPENCL  >> output.txt
done

for((i=problem_size;i<=100000000;i*=2));do
    bin/dotproduct $i CUDA  >> output.txt
done