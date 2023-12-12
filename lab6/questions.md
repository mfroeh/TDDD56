# How is the communication between the host and the graphic card handled?
- After identifying the device, a context for it is created
- This context is passed to the functions that require communication between GPU and host
- E.g. `createCommandQueue`, `createProgram`, `createBuffer` and implicitly when executing the kernel in `enqueueNDRangeKernel` through the previously created command queue

# What function executes your kernel?
- `clEnqueueNDRangeKernel`

# How does the kernel know what element to work on?
- Kernels can use the `get_global_id` function which returns the global id for a given dimension
- In our case, we only have one dimension so `get_global_id(0)` is sufficient
- The size is passed when running the kernel

# 
