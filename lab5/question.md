# How much data did you put in shared memory?
- (blockDim.y + 2 * kernelsizey) * (blockDim.x + 2 * kernelsizex)

# How much data does each thread copy to shared memory?
- Each thread copies (height / blockDim.y) * (width / blockDim.x)
- Scales negatively with the number of blocks -> large block dim less copies 

# How did you handle the necessary overlap between the blocks?
- Show sheet graphic for 2x2 block dim with 3x3 kernel

# If we would like to increase the block size, about how big blocks would be safe to use in this case? Why?
- As big as the GPU allows (32x32 for 1080)

# How much speedup did you get over the naive version? For what filter size?
- Naive: 113.879 (21x21 kernel img1.ppm 1 thread per block)
- Ours: 3.30346 (21x21 kernel img1.ppm 32x32 threads per block)

- Naive: 6.32717 (21x21 kernel img1.ppm 32x32 threads per block)
- Ours: 3.30346 (21x21 kernel img1.ppm 32x32 threads per block)

# Is your access to global memory coalesced? What should you do to get that?
- Not coalesced: A thread copies parts of multiple rows instead of copying one row at a time.
- Can make it coalesced by partitioning the data only row-wise, however, seems more complicated, as threads could e.g. do 2.3 rows

# How much speedup did you get over the non-separated? For what filter size?
- 3.02128 vs 3.3 (21x21 kernel img1.ppm 32x32 threads per block)

# Compare the visual result to that of the box filter. Is the image LP-filtered with the weighted kernel noticeably better?
- Average more blurry, gaussian more clear? Gaussian looks better

# What was the difference in time to a box filter of the same size (5x5)?
- Box: 1.6 ms, Gaussian: 2.1 ms (32x32)
- Gaussian: 2 * as many ops when computing pixel value (multiplication with gaussian weight)

# If you want to make a weighted kernel customizable by weights from the host, how would you deliver the weights to the GPU?
- cudaMalloc some memory, then cudaMemcpy the weights from host to that memory, then pass a pointer to the GPU memory to the kernel

# What kind of algorithm did you implement for finding the median?
- We count occurances of each of the 255 possible values for R,G,B each
- Then we go accumulate the values 0..=255
- Once we accumulate beyond half of the kernel region, we set the current value as the median

# What filter size was best for reducing noise?
- 5x5 too much
- 1-2x1-2 good

# Compare the result of the separable and full median filters.
- Result of seperable filter is much worse
- Output looks distorted

# Compare the difference in performance of the separable and full median filters.
- 121 seperable vs 1104 non-seperable (21x21 img1.ppm)