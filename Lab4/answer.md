*How many cores will simple.cu use, max, as written? How many SMs?*
1 SM, the program will open as many cores as needed dependent on the block size, which is 16 in this case => 1 SM. 
*Is the calculated square root identical to what the CPU calculates? Should we assume that this is always the case?*
No, that's not true. The GPU uses floats (single precision) and the CPU uses double (double preceision), so the numbers will be exact until the 16th decimal. 
*QUESTION: How do you calculate the index in the array, using 2-dimensional blocks?*
We can use the indexes and the dimensions of the blocks in the y and x directtions. Then the index of an 1D array in 2-dimensional block will be:
`` `c++`
    int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int elemIdx

``
*QUESTION: What happens if you use too many threads per block?*
The result is just 0s, unspecified behavior?

*At what data size is the GPU faster than the CPU?*
At N=64 the GPU is faster with blocksize 16

*What block size seems like a good choice? Compared to what?* 
16 worked good for us

*Write down your data size, block size and timing data for the best GPU performance you can get*
N=1024
blocksize = 16
time = 0.000059

*How much performance did you lose by making data accesses non-coalesced?*
Compared to previous non coalesced had timing of 0.000064.
And in total for our tests coalesced were about two times faster
*What were the main changes in order to make the Mandelbrot run in CUDA?*
We need to run the mandelbrot calculation on the GPU, so this should be a device kernel. For that to work we needed to make the cuComplex struct a device kernel too, and make the user controlled parameter managed memory.
In addition, the Draw() method is calling the kernel. 
*How many blocks and threads did you use?*
16x16 threads and 32x32 blocks.  
*When you use the Complex class, what modifier did you have to use on the methods?*
The struct should belong to the device, so (__device__) on every method that belongs to the struct.
*What performance did you get? How does that compare to the CPU solution?*
Roughly a 130x improvement if we use singleprecision.
*What performance did you get with float vs double precision?*
We can zoom a lot further with double. But the GPU time is just 30x times better than the CPU. 
*In Lab 1, load balancing was an important issue. Is that an issue here? Why/why not?*
No, loadbalancing should not be an issue on the GPU. Because each threads takes a point in the block/grid and the work is evently distributed over each thread the loadbalancing should not be an issue. 