/*
 * Placeholder OpenCL kernel
 */

__kernel void bitonic(__global unsigned int *data, int k, int j)
{ 
  unsigned int i = get_global_id(0);

  int ixj = i^j; 
  if((ixj) > i) {
    if((i & k) == 0 && (data[i]>data[ixj])) {
        unsigned int temp = data[i];
		    data[i] =data[ixj];
		    data[ixj] = temp;
    }
    if((i & k) != 0 && (data[i] < data[ixj])) {
        unsigned int temp = data[i];
		    data[i] =data[ixj];
		    data[ixj] = temp;
    }
  }

  //Something should happen here

}
