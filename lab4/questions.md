# How many cores will simple.cu use, max, as written? How many SMs?
- The grid is 1x1, meaning one block. That block is made up of 16x1=16 threads.
- Each block of threads is mapped to a single SM.
- Therefore at most 16 cores and 1 SM will be used.

# Is the calculated square root identical to what the CPU calculates? Should we assume that this is always the case?
- The square roots of floating point numbers in {x/2 : 0 <= x < 16} the cpu and gpu square roots are equal.
- TODO: Assume

# How do you calculate the index in the array, using 2-dimensional blocks?
- First we get our current blocks top-left coordinate using
    - blockIdx.x * dimBlock.x for the column and 
    - blockIdx.y * dimBlock.y for the row
- Then we just add our threads offset within the blocks
    - + threadIdx.x for the column and
    - + threadIdx.y for the row
- Note that this only works, if each thread is allocated exactly one element of the array (though I think that's what you do in CUDA?)

# What happens if you use too many threads per block?

# At what data size is the GPU faster than the CPU?

# What block size seems like a good choice? Compared to what?

# Write down your data size, block size and timing data for the best GPU performance you can get.
