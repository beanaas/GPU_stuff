*1.1*
En fördel med mapReduce är optimeringen av kod, t.ex hur vektorerna läggs i minnet och man slipper att skapa en separat vector V3. Det finns fler situationer där man kan optimera sin kod, t.ex loopfusion. 

*1.2* 
Ja, det finns fall när man vill separera map and reduce. T.ex när man har komplex funktioner för en map funktion men en enklare för reduce, så kan det vara värt att separera dessa och låt kompilatorn optimisera koden åt en.

*1.3*
We can see that OpenMP the slowest until the problem size is roughly 1/4 of a million, then the cpu will be the slowest. We can see that CUDA is always the fastest one, not depending on the problem size. 

*1.4*
Maybe there’s a big initialisation overhead with openMP? A lot of caching on the first run? 

*2.1* 
The separable version is the most efficient filter for us. The number of operations that are done are less with the separable filter, due to the area we average over is 2*radius for the separable filter and r^2 for the unified filter. 
So, instead of needing to take the average of the whole area we take the average row and column piecewise.

*3.1*
Yes, the processes are not dependent of each other. We just want to take the median of the pixels around.  Due to the sorting of the pixels, the median_kernels are independent of each other and the MapOverlap fits the task. 


*3.2*
We fill the user array with the neighbouring pixels, the we sort the array and return the median. 
The userfunction is data dependent due to us filling the array before we sort it. However this is not good for the GPU because we cannot vectorise it. 

