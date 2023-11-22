#define SKEPU_PRECOMPILED
#define SKEPU_OPENMP
/**************************
** TDDD56 Lab 3
***************************
** Author:
** August Ernstsson
**************************/

//#include "mapreduce.hpp"
#include <iostream>

#include <skepu>

/* SkePU user functions */

float multiply(float x, float y) { return x * y; }

float sum(float x, float y) { return x + y; }

// more user functions...


struct skepu_userfunction_skepu_skel_0mapReduceInstance_multiply
{
constexpr static size_t totalArity = 2;
constexpr static size_t outArity = 1;
constexpr static bool indexed = 0;
using IndexType = void;
using ElwiseArgs = std::tuple<float, float>;
using ContainerArgs = std::tuple<>;
using UniformArgs = std::tuple<>;
typedef std::tuple<> ProxyTags;
constexpr static skepu::AccessMode anyAccessMode[] = {
};

using Ret = float;

constexpr static bool prefersMatrix = 0;

#define SKEPU_USING_BACKEND_OMP 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block) block
#define VARIANT_CUDA(block)
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float OMP(float x, float y)
{ return x * y;
}
#undef SKEPU_USING_BACKEND_OMP

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float CPU(float x, float y)
{ return x * y;
}
#undef SKEPU_USING_BACKEND_CPU
};


struct skepu_userfunction_skepu_skel_0mapReduceInstance_sum
{
constexpr static size_t totalArity = 2;
constexpr static size_t outArity = 1;
constexpr static bool indexed = 0;
using IndexType = void;
using ElwiseArgs = std::tuple<float, float>;
using ContainerArgs = std::tuple<>;
using UniformArgs = std::tuple<>;
typedef std::tuple<> ProxyTags;
constexpr static skepu::AccessMode anyAccessMode[] = {
};

using Ret = float;

constexpr static bool prefersMatrix = 0;

#define SKEPU_USING_BACKEND_OMP 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block) block
#define VARIANT_CUDA(block)
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float OMP(float x, float y)
{ return x + y;
}
#undef SKEPU_USING_BACKEND_OMP

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float CPU(float x, float y)
{ return x + y;
}
#undef SKEPU_USING_BACKEND_CPU
};


struct skepu_userfunction_skepu_skel_1reduceInstance_sum
{
constexpr static size_t totalArity = 2;
constexpr static size_t outArity = 1;
constexpr static bool indexed = 0;
using IndexType = void;
using ElwiseArgs = std::tuple<>;
using ContainerArgs = std::tuple<>;
using UniformArgs = std::tuple<float, float>;
typedef std::tuple<> ProxyTags;
constexpr static skepu::AccessMode anyAccessMode[] = {
};

using Ret = float;

constexpr static bool prefersMatrix = 0;

#define SKEPU_USING_BACKEND_OMP 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block) block
#define VARIANT_CUDA(block)
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float OMP(float x, float y)
{ return x + y;
}
#undef SKEPU_USING_BACKEND_OMP

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float CPU(float x, float y)
{ return x + y;
}
#undef SKEPU_USING_BACKEND_CPU
};


struct skepu_userfunction_skepu_skel_2mapInstance_multiply
{
constexpr static size_t totalArity = 2;
constexpr static size_t outArity = 1;
constexpr static bool indexed = 0;
using IndexType = void;
using ElwiseArgs = std::tuple<float, float>;
using ContainerArgs = std::tuple<>;
using UniformArgs = std::tuple<>;
typedef std::tuple<> ProxyTags;
constexpr static skepu::AccessMode anyAccessMode[] = {
};

using Ret = float;

constexpr static bool prefersMatrix = 0;

#define SKEPU_USING_BACKEND_OMP 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block) block
#define VARIANT_CUDA(block)
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float OMP(float x, float y)
{ return x * y;
}
#undef SKEPU_USING_BACKEND_OMP

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float CPU(float x, float y)
{ return x * y;
}
#undef SKEPU_USING_BACKEND_CPU
};

int main(int argc, const char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <input size> <backend>\n";
    exit(1);
  }

  const size_t size = std::stoul(argv[1]);
  auto spec = skepu::BackendSpec{argv[2]};
  //	spec.setCPUThreads(<integer value>);
  skepu::setGlobalBackendSpec(spec);

  /* Skeleton instances */
  skepu::backend::Map<2, skepu_userfunction_skepu_skel_2mapInstance_multiply, bool, void> mapInstance(false);
  skepu::backend::Reduce1D<skepu_userfunction_skepu_skel_1reduceInstance_sum, bool, void> reduceInstance(false);
  skepu::backend::MapReduce<2, skepu_userfunction_skepu_skel_0mapReduceInstance_multiply, skepu_userfunction_skepu_skel_0mapReduceInstance_sum, bool, bool, void> mapReduceInstance(false, false);

  /* SkePU containers */
  skepu::Vector<float> v1(size, 1.0f), v2(size, 2.0f), vMapRes(size);

  /* Compute and measure time */
  float resComb, resSep;

  auto timeComb = skepu::benchmark::measureExecTime([&] {
    resComb = mapReduceInstance(v1, v2);
  });

  auto timeSep = skepu::benchmark::measureExecTime([&] {
    // your code here
    mapInstance(vMapRes, v1, v2);
    resSep = reduceInstance(vMapRes);
  });

  std::cout << "Time Combined: " << (timeComb.count() / 10E6) << " seconds.\n";
  std::cout << "Time Separate: " << (timeSep.count() / 10E6) << " seconds.\n";

  std::cout << "Result Combined: " << resComb << "\n";
  std::cout << "Result Separate: " << resSep << "\n";

  return 0;
}
