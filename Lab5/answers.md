QUESTION: How much data did you put in shared memory?

[BLOCK_SIZE+2*maxKernelSizeX][(BLOCK_SIZE+2*maxKernelSizeY)*3]

QUESTION: How much data does each thread copy to shared memory?

It depends on how large the kernel is.

QUESTION: How did you handle the necessary overlap between the blocks?

By dividing the picture into block then adding overlap inside the filter. Other implementation could be to include overlap before passing to boxfilter, meaning one thread per pixel.

QUESTION: If we would like to increase the block size, about how big blocks would be safe to use in this case? Why?

Now the block size could be max 32 since the overlap is added inside the filter. 32x32=1024 which is max

QUESTION: How much speedup did you get over the naive version? For what filter size?
7x7 kernelsize roughly 100x faster. 
Naive: 0.01942 Shared memory: 0.00203
QUESTION: Is your access to global memory coalesced? What should you do to get that?

No, since 32 is not devicable by 3. 

QUESTION: How much speedup did you get over the non-separated? For what filter size?
Kernelsize 7x7c roughly 2x faster. 
Non-separated: 0.000204
Seperable: 0.000086

QUESTION: Compare the visual result to that of the box filter. Is the image LP-filtered with the weighted kernel noticeably better?
The gaussion filter we would say is noticeably better at resembling the original image

QUESTION: What was the difference in time to a box filter of the same size (5x5)?

QUESTION: If you want to make a weighted kernel customizable by weights from the host, how would you deliver the weights to the GPU?

Use __managed__ so it is available to the host and the device

QUESTION: What kind of algorithm did you implement for finding the median?

Histogram

QUESTION: What filter size was best for reducing noise?
