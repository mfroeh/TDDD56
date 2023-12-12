/*
 * Placeholder OpenCL kernel
 */

__kernel void find_max(__global unsigned int *data, const unsigned int length)
{ 
  unsigned int gid = get_global_id(0);
  unsigned int lid = get_local_id(0);

  __local unsigned local_mem[256];
  local_mem[lid] = data[gid];
  unsigned lsize = get_local_size(0); 
  unsigned gsize = get_global_size(0);


  barrier(CLK_LOCAL_MEM_FENCE);

  
    for (unsigned int s=1; s < lsize; s *= 2) {
        int index = 2 * s * lid;
        if (index < lsize) {
          local_mem[index] = max(local_mem[index + s], local_mem[index]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0){
      //printf("GID: %d \n", gid);
      data[gid] = local_mem[0];
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    if(gid == 0){
      unsigned maximum = data[0];

      for(unsigned int i = 0; i < gsize; i += lsize)
      {
        printf("I: %d \n", i);
        if(maximum < data[i])
          maximum = data[i];
      }

      data[0] = maximum;
    }

}

