#!/bin/bash

number_threads=8
echo "POP MEASURE LOCK">>output.csv
for((i=1;i<=number_threads;i++));do
    make clean; make MEASURE=1 NON_BLOCKING=0 NB_THREADS=$i
    echo "thread$i">>output.csv
    ./stack >> output.csv
done
echo "POP MEASURE CAS">>output.csv
for((i=1;i<=number_threads;i++));do
    make clean; make MEASURE=1 NON_BLOCKING=1 NB_THREADS=$i
    echo "thread$i">>output.csv
    ./stack >> output.csv
done

echo "PUSH MEASURE LOCK">>output.csv
for((i=1;i<=number_threads;i++));do
    make clean; make MEASURE=2 NON_BLOCKING=0 NB_THREADS=$i
    echo "thread$i">>output.csv
    ./stack >> output.csv
done

echo "PUSH MEASURE CAS">>output.csv
for((i=1;i<=number_threads;i++));do
    make clean; make MEASURE=2 NON_BLOCKING=1 NB_THREADS=$i
    echo "thread$i">>output.csv
    ./stack >> output.csv
done