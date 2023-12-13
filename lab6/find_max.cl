__kernel void find_max(__global unsigned *data, unsigned s) {
  int index = get_global_id(0);
  if (data[index + s] > data[index]) {
    data[index] = data[index + s];
  }
}
