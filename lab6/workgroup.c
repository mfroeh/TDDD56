#include <stdio.h>
#include <CL/cl.h>

int main(){

    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    size_t workgroup_size;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &workgroup_size, NULL);

    printf("Max work group size: %zu\n", workgroup_size);

    return 0;
}