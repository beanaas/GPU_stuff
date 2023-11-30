#!/bin/bash

rm "output_q31.txt"
#make clean; make bin/average
size=8
for((i=2;i<=size;i*=2));do
    bin/median data/small.png testing2 $i CPU  >> output_q31.txt
    echo $i
done
echo "done"
for((i=2;i<=size;i*=2));do
    bin/median data/small.png testing2 $i OPENMP  >> output_q31.txt
    echo $i
done
echo "done"
for((i=2;i<=size;i*=2));do
    bin/median data/small.png testing2 $i OPENCL  >> output_q31.txt
    echo $i
done
echo "done"
for((i=2;i<=size;i*=2));do
    bin/median data/small.png testing2 $i CUDA  >> output_q31.txt
    echo $i
done