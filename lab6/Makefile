hello_world_cl: hello_world_cl.c CLutilities.c
	gcc hello_world_cl.c CLutilities.c -lOpenCL -o hello_world_cl

find_max: find_max.c find_max.cl milli.c CLutilities.c
	gcc find_max.c milli.c CLutilities.c -lOpenCL -o find_max
