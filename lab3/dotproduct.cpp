/**************************
** TDDD56 Lab 3
***************************
** Author:
** August Ernstsson
**************************/

//#include "mapreduce.hpp"
#include <iostream>

#include <skepu>

float add(float lhs, float rhs) { return lhs + rhs; }
float mul(float lhs, float rhs) { return lhs * rhs; }

int main(int argc, const char* argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <input size> <backend>\n";
    exit(1);
  }

  const size_t size = std::stoul(argv[1]);
  auto spec = skepu::BackendSpec{argv[2]};
  //	spec.setCPUThreads(<integer value>);
  skepu::setGlobalBackendSpec(spec);

  /* Skeleton instances */
  auto map{skepu::Map<2>(mul)};
  auto reduce{skepu::Reduce(add)};
  auto mapreduce{skepu::MapReduce<2>(mul, add)};

  /* SkePU containers */
  skepu::Vector<float> v1(size, 1.0f), v2(size, 2.0f);

  /* Compute and measure time */
  float resComb, resSep;
  auto timeComb = skepu::benchmark::measureExecTime([&] { resComb = mapreduce(v1, v2); });

  auto timeSep = skepu::benchmark::measureExecTime([&] {
    skepu::Vector<float> mapres(v1.size());
    map(mapres, v1, v2);
    resSep = reduce(mapres);
  });

  auto timeCombCold = skepu::benchmark::measureExecTimeIdempotent([&] {
	resComb = mapreduce(v1, v2); 
  });

  auto timeSepCold = skepu::benchmark::measureExecTimeIdempotent([&] {
    skepu::Vector<float> mapres(v1.size());
    map(mapres, v1, v2);
    resSep = reduce(mapres);
  });

  std::cout << "Time Combined: " << (timeComb.count() / 10E6) << " seconds.\n";
  std::cout << "Time Separate: " << (timeSep.count() / 10E6) << " seconds.\n";

  std::cout << "Time Combined (cold run): " << (timeCombCold.count() / 10E6) << " seconds.\n";
  std::cout << "Time Separate (cold run): " << (timeSepCold.count() / 10E6) << " seconds.\n";

  std::cout << "Result Combined: " << resComb << "\n";
  std::cout << "Result Separate: " << resSep << "\n";

  return 0;
}
