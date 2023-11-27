*How many cores will simple.cu use, max, as written? How many SMs?*
1 SM, the program will open as many cores as needed dependent on the block size, which is 16 in this case => 1 SM. 
*Is the calculated square root identical to what the CPU calculates? Should we assume that this is always the case?*
