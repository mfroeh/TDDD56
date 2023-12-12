/*
 * Placeholder OpenCL kernel
 */

__kernel void find_max(__global unsigned int *data, const unsigned int length)
{ 
  unsigned int pos = 0;
  unsigned int val = 0;

  size_t off = get_global_id(0);
  val = max(val, data[pos + off]);

  data[0] = max(data[0], val);
}
