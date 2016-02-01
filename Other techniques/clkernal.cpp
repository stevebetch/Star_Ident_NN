# include "clkernal.h"
#include <libc.h>
#include <stdbool.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <OpenCL/opencl.h>

#include <math.h>

#define PADDING         (32)
#define GROUP_DIMX      (32)
#define LOG_GROUP_DIMX  (5)
#define GROUP_DIMY      (2)

#define WIDTH           (256)
#define HEIGHT          (4096)

static int iterations = 100;
static int width      = 256;
static int height     = 4096;


int moo(long double matrix, long double x, long double y, int col, int row)
{
//    uint64_t         t0, t1, t2;
    int              err;
    cl_device_id     device_id;
    cl_context       context;
    cl_kernel        kernel;
    cl_command_queue queue;
    cl_program       program;
    cl_mem			 a_in, b_in, c_out;
    
    size_t global[2], local[2];
    global[0] = width * GROUP_DIMY;
    global[1] = height / GROUP_DIMX;
    local[0] = GROUP_DIMX * GROUP_DIMY;
    local[1] = 1;
    
    // Connect to a GPU compute device
    //
    int gpu = 1;
    err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }
    
    // Create a compute context
    //
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }
    
    // Create a command queue
    //
    queue = clCreateCommandQueue(context, device_id, 0, &err);
    if (!queue)
    {
        printf("Error: Failed to create a command queue!\n");
        return EXIT_FAILURE;
    }
    
    // Load the compute program from disk into a cstring buffer
    //
    
    char *source = "mathkernl.cl";
   
    
    // Create the compute program from the source buffer
    //
    program = clCreateProgramWithSource(context, 1, (const char **) & source, NULL, &err);
    if (!program || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }
    
    // Build the program executable
    //
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];
        
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return EXIT_FAILURE;
    }
    
    
    // Setup memory objects, assume square matrix....
    //long double kern_mat[col*row], kern_x[col], kern_out[col];
    
    a_in=clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(long double)*col*row, NULL, NULL);
    b_in=clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(long double)*col, NULL, NULL);
    c_out=clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(long double)*col, NULL, NULL);
    
    
    
    // Create the compute kernel from within the program
    //
    kernel = clCreateKernel(program, "matvec", &err);
    if (!kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        return EXIT_FAILURE;
    }

    err=0;
    //attach arguments to the kernal function
    err=clSetKernelArg(kernel,0,sizeof(cl_mem),&a_in);
    err|=clSetKernelArg(kernel,1,sizeof(cl_mem),&b_in);
    err|=clSetKernelArg(kernel,2,sizeof(cl_mem),&c_out);
    err|=clSetKernelArg(kernel,3,sizeof(int),&col);
    
    if (err)
    {
        printf("Error: Failed to attach arguments to the kernal function!\n");
        return EXIT_FAILURE;
    }
    //write buffers from host to global mem
    err=0;
    err=clEnqueueWriteBuffer(queue, a_in, CL_FALSE, 0, sizeof(long double)*col*row, &matrix, 0, NULL, NULL);
    err=clEnqueueWriteBuffer(queue, a_in, CL_FALSE, 0, sizeof(long double)*col, &x, 0, NULL, NULL);
    
    if (err)
    {
        printf("Error: Failed to write to buffer!\n");
        return EXIT_FAILURE;
    }
    
    err=0;
    err=clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }
     err=0;
    err= clEnqueueReadBuffer(queue, c_out, CL_TRUE, 0, sizeof(long double)*col, &y, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to read back results from the device!\n");
        return EXIT_FAILURE;
    }
    
    clReleaseMemObject(a_in);
    clReleaseMemObject(c_out);
    clReleaseMemObject(b_in);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    
    
    
    return 0;
    
}