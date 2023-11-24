CC = nvcc
CC_FLAGS = -std=c++20

simple: simple.cu
	$(CC) $(CC_FLAGS) simple.cu -o simple

maddgpu: matrix_gpu.cu
	$(CC) $(CC_FLAGS) matrix_gpu.cu -o maddgpu

maddcpu: matrix_cpu.cc
	g++ matrix_cpu.cc -o maddcpu -std=c++20

clean:
	rm -f simple maddcpu maddgpu
