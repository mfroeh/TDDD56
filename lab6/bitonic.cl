inline void exchange(unsigned *i, unsigned *j) {
  int tmp = *i;
  *i = *j;
  *j = tmp;
}

__kernel void bitonic(__global unsigned *data, unsigned k, unsigned j) {
  unsigned i = get_global_id(0);
  int ixj = i ^ j;

  if (ixj > i) {
    // Thread in here got work for this minor iteration
    if ((i & k) == 0 && data[i] > data[ixj]) {
      exchange(&data[i], &data[ixj]);
    }
    if ((i & k) != 0 && data[i] < data[ixj]) {
      exchange(&data[i], &data[ixj]);
    }
  }
}
