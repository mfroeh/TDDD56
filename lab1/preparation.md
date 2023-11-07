# 1. 
The complex numbers that are in the mandelbrot set, are those for which the Julia sequence converges. 
We cannot algorithmically detect convergence, but we can detect divergence by simply computing the Julia sequence.
We color those pixels black, for which we could not detect divergence within `max_iter` iterations.
If we detect divergence before `max_iter` iterations, we can stop.
Therefore black pixels take longer to compute than non-black pixels.
By looking at an image of the mandelbrot set, we can see that the distribution of converging and diverging points is not uniform.
E.g. if a thread is assigned only converging points, it will take longer than a thread that is assigned a mix of diverging and converging points.

The threads in the center (thread 1 and 2 in Figure 4) are assigned more converging points than those outside the center.
Especially for thread counts $2n+1, n > 0$, the center thread will always have the most work.

# 2.
- We used static cyclic load balancing with a row-wise partitioning where thread $n < NB_THREADS$ is assigned rows $n, n+NB_THREADS, n+2*NB_THREADS..., $.
- We used dynamic, task based load balancing with a global shared work pool. With worker threads autonomously fetching tasks from the work pool using an atomically incremented counter variable. We used the atomic add_and_fetch instruction, to avoid using mutex locks.

# 3. 

# 4.
Use atomic instructions if available.
We could also avoid using a shared work pool, as it is implicitly encoded in the counter variable.

# 5.
Does not generalize well. We did not implement an abstraction for a task runtime system and instead have workers grabbing their tasks. Furthermore, workers are not suspended but are alive for as long as they can fetch tasks (actually good for performance, but not as general).
