## How is the communication between the host and the graphic card handled?
- After identifying the device, a context for it is created
- This context is passed to the functions that require communication between GPU and host
- E.g. `createCommandQueue`, `createProgram`, `createBuffer` and implicitly when executing the kernel in `enqueueNDRangeKernel` through the previously created command queue

## What function executes your kernel?
- `clEnqueueNDRangeKernel`

## How does the kernel know what element to work on?
- Kernels can use the `get_global_id` function which returns the global id for a given dimension
- In our case, we only have one dimension so `get_global_id(0)` is sufficient
- The size is passed when running the kernel

## What timing did you get for your GPU reduction? Compare it to the CPU version.
- CPU is slower for N <= 2^16 and slower than GPU from then on
- N = 2^16: CPU 0.000203 vs GPU 0.000441
- N = 2^17: CPU 0.000823 vs GPU 0.000813
- N = 2^18: CPU 0.001154 vs GPU 0.000744
- N = 2^28: CPU 0.468195 vs GPU 0.208603

## Try larger data size. On what size does the GPU version get faster, or at least comparable, to the CPU?
- N = 2^16: CPU 0.000203 vs GPU 0.000441
- N = 2^17: CPU 0.000823 vs GPU 0.000813
- N = 2^18: CPU 0.001154 vs GPU 0.000744
- N = 2^28: CPU 0.468195 vs GPU 0.208603

## How can you optimize this further? You should know at least one way.
- Local memory, but we use that already
- 

## Should each thread produce one output or two? Why?
- In a given minor stage (loop over j) a thread, if it gets work, will swap either two elements or none

## How many items can you handle in one workgroup?
- 1024

## What problem must be solved when you use more than one workgroup? How did you solve it?
- In OpenCL there is no synchronization between work groups, therefore we have to call our kernel function multiple times
- In CUDA, we may have had a different solution where we move the major and minor loop into the kernel as well

## What time do you get? Difference to the CPU? What is the break even size? What can you expect for a parallel CPU version? (Your conclusions here may vary between the labs.)
- N = 2^11: CPU 0.000960 vs GPU 0.000924
- N = 2^12: CPU 0.002162 vs GPU 0.001670
- N = 2^13: CPU 0.003157 vs GPU 0.001372
- N = 2^26: CPU 76.180239 vs GPU 0.740843

- Inner most loop is trivially parallelizable, so we should get a speedup that is close the amount of logical cores on the CPU
- In addition, as long as logical cores > N, we don't run into the "problem" of some threads not having any work in some of the minor iterations
