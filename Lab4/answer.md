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
