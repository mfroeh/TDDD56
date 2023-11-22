# Question 1.1: Why does SkePU have a "fused" MapReduce when there already are separate Map and Reduce skeletons?
- No need for barrier synchronization after Map is done (before Reduce could start)
- The data that will be used for the Reduce operation is in the cache of the core executing the Map already when MapReduce is used.
  If Map and Reduce were to be used in succession, it's unlikely that the same cores that did the Map, do the corresponding reduce. 
  Thus there would be cache misses at the Reduce step.

# Question 1.2: Is there any practical reason to ever use separate Map and Reduce in sequence?

# Question 1.3: Is there a SkePU backend which is always more efficient to use, or does this depend on the problem size? Why? Either show with measurements or provide a valid reasoning. 
- The optimal backend is highly dependend on the problem size.
- Small problems (e.g. naive sum of 1-100), the overhead of using the OpenMP (thread creation and barrier synchronization) or GPU backends (moving data to GPU) would outscale the gain from parallelization.
- Problems that involve large amounts of data (e.g. image filters) will generally perform much better on the GPU backends, due to their far larger core count.

# Question 1.4: Try measuring the parallel backends with measureExecTime exchanged for measureExecTimeIdempotent. This measurement does a "cold run" of the lambda expression before running the proper measurement. Do you see a difference for some backends, and if so, why?

# Question 2.1: Which version of the averaging filter (unified, separable) is the most efficient? Why?

# Question 3.1: In data-parallel skeletons like MapOverlap, all elements are processed independently of each other. Is this a good fit for the median filter? Why/why not?

# Question 3.2: Describe the sequence of instructions executed in your userfunction. Is it data dependent? What does this mean for e.g., automatic vectorization, or the GPU backend?


