__kernel void find_max(__global unsigned *data, unsigned length) {
  unsigned gid = get_global_id(0);
  unsigned lid = get_local_id(0);
  unsigned lsize = get_local_size(0);
  unsigned gsize = get_global_size(0);

  // Copy to local (shared) memory
  __local unsigned local_mem[1024];
  local_mem[lid] = data[gid];
  barrier(CLK_LOCAL_MEM_FENCE);

  // Interleaved addressing on local (shared) memory
  // With increasing stride, fewer and fewer threads get work
  for (unsigned int s = 1; s < lsize; s *= 2) {
    barrier(CLK_LOCAL_MEM_FENCE);
    int index = 2 * s * lid;
    if (index < lsize) {
      if (local_mem[index + s] > local_mem[index]) {
        local_mem[index] = local_mem[index + s];
      }
    }
  }

  // First local thread writes back local max
  if (lid == 0) {
    unsigned wb = gid / lsize;
    data[wb] = local_mem[0];
  }
}
