filter: filtercu milli.c readppm.c
	g++ filter.o milli.c readppm.c -lGL -lglut -lm -lcuda -lcudart -L/usr/local/cuda/lib64 -o filter

# TODO: -arch=sm_30 in uni rooms
filtercu: filter.cu
	nvcc filter.cu -c -o filter.o

clean:
	rm *.o
	rm filter
