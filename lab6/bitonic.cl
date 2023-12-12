static void exchange(unsigned int *i, unsigned int *j) {
  int k;
  k = *i;
  *i = *j;
  *j = k;
}

__kernel void bitonic(__global unsigned int *data, const unsigned int length)
{ 
 unsigned int i, j, k;
  unsigned N = length;

  for (k = 2; k <= N; k = 2 * k)  // Outer loop, double size for each step
  {
    for (j = k >> 1; j > 0; j = j >> 1)  // Inner loop, half size for each step
    {
      for (i = 0; i < N; i++)  // Loop over data
      {
        int ixj = i ^ j;  // Calculate indexing!
        if ((ixj) > i) {
          if ((i & k) == 0 && data[i] > data[ixj]) {
            exchange(&data[i], &data[ixj]);
          }
          if ((i & k) != 0 && data[i] < data[ixj]) {
            exchange(&data[i], &data[ixj]);
          }
        }
      }
    }
  }
}
