#!/bin/bash

number_threads=8
rm "output.txt"
#echo "POP MEASURE LOCK">>output.txt
for((i=1;i<=number_threads;i++));do
    make clean; make MEASURE=1 NON_BLOCKING=0 NB_THREADS=$i MAX_PUSH_POP=1000000
    #echo "thread$i">>output.txt
    ./stack >> output.txt
done
#echo "POP MEASURE CAS">>output.txt
for((i=1;i<=number_threads;i++));do
    make clean; make MEASURE=1 NON_BLOCKING=1 NB_THREADS=$i MAX_PUSH_POP=1000000
    #echo "thread$i">>output.txt
    ./stack >> output.txt
done

#echo "PUSH MEASURE LOCK">>output.txt
for((i=1;i<=number_threads;i++));do
    make clean; make MEASURE=2 NON_BLOCKING=0 NB_THREADS=$i MAX_PUSH_POP=1000000
    #echo "thread$i">>output.txt
    ./stack >> output.txt
done

#echo "PUSH MEASURE CAS">>output.txt
for((i=1;i<=number_threads;i++));do
    make clean; make MEASURE=2 NON_BLOCKING=1 NB_THREADS=$i MAX_PUSH_POP=1000000
    #echo "thread$i">>output.txt
    ./stack >> output.txt
done