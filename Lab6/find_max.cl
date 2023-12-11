/*
 * Placeholder OpenCL kernel

 \/ \/ \/ \/
  \/ \/ \/
   \/  \/
     \/
 */

__kernel void find_max(__global unsigned int *data, const unsigned int length)
{ 
  unsigned int val;
  unsigned int threadIdX = get_global_id(0);
  
  val = max(data[threadIdX*2+ 0], data[threadIdX*2 + 1]);  //find the maximum of the 2 values
  barrier(CLK_LOCAL_MEM_FENCE); //SYNC
  //Write
  data[get_global_id(0)] = val;
}
