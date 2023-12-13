# How many cores will simple.cu use, max, as written? How many SMs?
- The grid is 1x1, meaning one block. That block is made up of 16x1=16 threads.
- Each block of threads is mapped to a single SM.
- Therefore at most 16 cores and 1 SM will be used.

# Is the calculated square root identical to what the CPU calculates? Should we assume that this is always the case?
- The square roots of floating point numbers in {x/2 : 0 <= x < 16} the cpu and gpu square roots are equal.
- If different floating point precision is used on CPU and GPU, then the result could differ

# How do you calculate the index in the array, using 2-dimensional blocks?
- First we get our current blocks top-left coordinate using
    - blockIdx.x * dimBlock.x for the column and 
    - blockIdx.y * dimBlock.y for the row
- Then we just add our threads offset within the blocks
    - + threadIdx.x for the column and
    - + threadIdx.y for the row
- Note that this only works, if each thread is allocated exactly one element of the array (though I think that's what you do in CUDA?)

# What happens if you use too many threads per block?
- We get a CUDA error, for the 1080 a maximum 1024 are allowed per block

# At what data size is the GPU faster than the CPU?
- At N = 64 GPU is faster than CPU

# What block size seems like a good choice? Compared to what?
- The ideal block size is the maximum allowed block size (32 x 32)
- 32x32 is twice as fast as blocks of 2x2

# Write down your data size, block size and timing data for the best GPU performance you can get.
- For 1024 * 2^4: 14 ms on GPU vs 11540ms on CPU
- Block size: 32x32 since optimal

# How much performance did you lose by making data accesses non-coalesced?
2048	0.26656	0.582752	2.186194478
4096	0.900032	2.34384	2.604174074
8192	3.77094	11.1317	2.951969535
- Becomes more important the larger N grows

# What were the main changes in order to make the Mandelbrot run in CUDA?
- Make mandelbrot a kernel
- Compute our grid value (x, y)
- Malloc memory on GPU and memcpy from GPU memory to host memory

# How many blocks and threads did you use?
- 1024/32 x 1024/32 grid dim
- 32x32 block dim

# When you use the Complex class, what modifier did you have to use on the methods?
- Make all functions in cuComplex __device__

# What performance did you get? How does that compare to the CPU solution?
- 939 ms vs 1 ms on average for DIM=1024 maxiter=200 4byte precision

# What performance did you get with float vs double precision?
- 25895 ms vs 4 ms on average for DIM=1024 maxiter=200 8byte precision

# In Lab 1, load balancing was an important issue. Is that an issue here? Why/why not?
- We have one thread per pixel and can run all those pixels in parallel, in the CPU lab we only had 16 logical cores
- If one block takes longer than the rest, the GPU scheduler will simply schedule on SMs that are free