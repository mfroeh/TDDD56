filter: filtercu milli.c readppm.c
	g++ filter.o milli.c readppm.c -lGL -lglut -lm -lcuda -lcudart -L/usr/local/cuda/lib64 -o filter

naive: naivecu milli.c readppm.c
	g++ naive.o milli.c readppm.c -lGL -lglut -lm -lcuda -lcudart -L/usr/local/cuda/lib64 -o naive

# TODO: -arch=sm_30 in uni rooms
filtercu: filter.cu
	nvcc filter.cu -c -o filter.o

naivecu: naive.cu
	nvcc naive.cu -c -o naive.o

clean:
	rm *.o
	rm filter
