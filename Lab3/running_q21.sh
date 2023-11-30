#!/bin/bash

rm "output_q21.txt"
#make clean; make bin/average
size=32
for((i=2;i<=size;i*=2));do
    bin/average data/small.png testing $i CPU  >> output_q21.txt
done
echo "done"
for((i=2;i<=size;i*=2));do
    bin/average data/small.png testing $i OPENMP  >> output_q21.txt
done
echo "done"
for((i=2;i<=size;i*=2));do
    bin/average data/small.png testing $i OPENCL  >> output_q21.txt
done
echo "done"
for((i=2;i<=size;i*=2));do
    bin/average data/small.png testing $i CUDA  >> output_q21.txt 
done