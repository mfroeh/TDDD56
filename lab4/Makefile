CC = nvcc
CC_FLAGS = -std=c++14

simple: simple.cu
	$(CC) $(CC_FLAGS) simple.cu -o simple

maddgpu: matrix_gpu.cu
	$(CC) $(CC_FLAGS) matrix_gpu.cu -o maddgpu 

maddcpu: matrix_cpu.cc
	g++ matrix_cpu.cc milli.c -o maddcpu -std=c++14

mandel: interactiveMandelbrot.cu
	$(CC) $(CC_FLAGS) -O3 interactiveMandelbrot.cu -o mandel -lglut -lGL

mandelcpu: interactiveMandelbrot.cpp
	g++ interactiveMandelbrot.cpp milli.c -o mandelcpu -lglut -lGL

clean:
	rm -f simple maddcpu maddgpu
