//
//  main.cpp
//  eigentry
//
//  Created by Stephan Boettcher on 4/25/13.
//  Copyright (c) 2013 Stephan Boettcher. All rights reserved.
//
#include <iostream>
#include <fstream>
#include <OpenCL/OpenCL.h>
#include <OpenCL/cl.h>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <cilk.h>
#include <reducer_opadd.h>
#include <string>
#include <libc.h>
#include <stdbool.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <math.h>
#include <unistd.h>




#include "clkernal.h"

#define PADDING         (32)
#define GROUP_DIMX      (32)
#define LOG_GROUP_DIMX  (5)
#define GROUP_DIMY      (2)
//
#define WIDTH           (1)
#define HEIGHT          (300)
//
//static int iterations = 100;
static int width      = 256;
static int height     = 4096;



using namespace std;

long double rand( long double a, long double b)
{
    
    long double debugger=(rand()*fabs(b-a))/RAND_MAX +a;
    return debugger;
};

static char * load_program_source(const char *filename);
#define size 300

/* =================================================== */
/*
 * Timing functions
 */
#if !defined(HAVE_TIMER)
#  define TIMER_DESC "gettimeofday"

#define USE_STD_CREATE
#define USE_STD_DESTROY

#include <sys/time.h>

struct stopwatch_t * stopwatch_create (void);
void stopwatch_destroy (struct stopwatch_t* T);
void stopwatch_init (void);

void stopwatch_start (struct stopwatch_t* T);

long double stopwatch_stop (struct stopwatch_t* T);




struct stopwatch_t
{
    struct timeval t_start_;
    struct timeval t_stop_;
    int is_running_;
};

static
long double
elapsed (struct timeval start, struct timeval stop)
{
    return (long double)(stop.tv_sec - start.tv_sec)
    + (long double)(stop.tv_usec - start.tv_usec)*1e-6;
}

long double
stopwatch_elapsed (struct stopwatch_t* T)
{
    long double dt = 0;
    if (T) {
        if (T->is_running_) {
            struct timeval stop;
            gettimeofday (&stop, 0);
            dt = elapsed (T->t_start_, stop);
        } else {
            dt = elapsed (T->t_start_, T->t_stop_);
        }
    }
    return dt;
}

void
stopwatch_init (void)
{
    fprintf (stderr, "Timer: %s\n", TIMER_DESC);
    fprintf (stderr, "Timer resolution: ~ 1 us (?)\n");
    fflush (stderr);
}

void
stopwatch_start (struct stopwatch_t* T)
{
    assert (T);
    T->is_running_ = 1;
    gettimeofday (&(T->t_start_), 0);
}

long double
stopwatch_stop (struct stopwatch_t* T)
{
    long double dt = 0;
    if (T) {
        if (T->is_running_) {
            gettimeofday (&(T->t_stop_), 0);
            T->is_running_ = 0;
        }
        dt = stopwatch_elapsed (T);
    }
    return dt;
}

#  define HAVE_TIMER 1
#endif

#if defined(USE_STD_CREATE)
struct stopwatch_t *
stopwatch_create (void)
{
    struct stopwatch_t* new_timer =
    (struct stopwatch_t *)malloc (sizeof (struct stopwatch_t));
    if (new_timer)
        memset (new_timer, 0, sizeof (struct stopwatch_t));
    return new_timer;
}
#endif

#if defined(USE_STD_DESTROY)
void
stopwatch_destroy (struct stopwatch_t* T)
{
    if (T) {
        stopwatch_stop (T);
        free (T);
    }
}
#endif
/* =================================================== */

int main(int argc, const char * argv[])
{
    int itters =1000;
   

    double sequentala[size][size];
    double sequentalb[size];
    double sequentalc[size];
    double cilka[size][size];
    double cilkb[size];
    double cilkc[size];

    double gpua[size*size];
    double gpub[size];
   double gpuc[size];
    
    
    for(int i=0;i<size;i++)
    {
        for(int j=0;j<size;j++){
            
            sequentala[i][j]=rand(-5,5);
            cilka[i][j]=sequentala[i][j];
            gpua[i*size+j]=sequentala[i][j];
        }
        sequentalb[i]=rand(-5,5);;
        cilkb[i]=sequentalb[i];
        gpub[i]=sequentalb[i];
        
        sequentalc[i]=0;
        cilkc[i]=0;
        gpuc[i]=0;
        
    }
    
///////////////// SEQUENTIAL ARRAY MULT///////////
    
    // timer
    struct stopwatch_t* tseq = NULL;
    long double  t_seq;
    // initialize timer
    stopwatch_init ();
    tseq = stopwatch_create ();
    stopwatch_start (tseq); //start timer
    
    for (int k=0;k<itters;k++){
        
    
    for (int i=0; i<size; i++) {
        
        for (int j=0; j<size; j++) {
            sequentalc[i]+=sequentala[i][j]*sequentalb[i];
        }
    }
        
    }
   t_seq = stopwatch_stop (tseq);
    fprintf (stderr, "Time to execute squential mult: %Lg secs\n",t_seq);

    ///////////////// cilk ARRAY MULT///////////
    
    // timer
    struct stopwatch_t* tcilk = NULL;
    long double  t_cilk;
    // initialize timer
    stopwatch_init ();
    tcilk = stopwatch_create ();
    stopwatch_start (tcilk); //start timer
    
    for (int k=0;k<itters;k++){
    cilk_for (int i=0; i<size; i++) {
        cilk::reducer_opadd<double> mooer(0.0);
        cilk_for (int j=0; j<size; j++) {
            mooer=mooer+cilka[i][j]*cilkb[i];
        }
        cilkc[i]=mooer.get_value();
    }
        
    }
    t_cilk= stopwatch_stop (tcilk);
    fprintf (stderr, "Time to execute cilk mult: %Lg secs\n",t_cilk);
    
    ///////////////// GPU ARRAY MULT///////////
 
    // timer
    struct stopwatch_t* tgpu = NULL;
    long double  t_tgpu;
    // initialize timer
    stopwatch_init ();
    tgpu = stopwatch_create ();
    stopwatch_start (tgpu); //start timer
    for(int moo23=0;moo23<itters;moo23++)
    {
    ///..///..///...//.././././..
  
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
    const char* filename = "/Users/stephan/xcode/eigentry/eigentry/mathkernl.cl";
    char *source = load_program_source(filename);
    
    if(!source)
    {
        printf("Error: Failed to load compute program from file!\n");
        return EXIT_FAILURE;
    }

    
    // Create the compute program from the source buffer
    //
    program = clCreateProgramWithSource(context, 1, (const char **) &source, NULL, &err);
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
    //long double kern_mat[size*size], kern_x[size], kern_out[size];
    
    a_in=clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double)*size*size, NULL, NULL);
    b_in=clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double)*size, NULL, NULL);
    c_out=clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double)*size, NULL, NULL);
    
    
    
    // Create the compute kernel from within the program
    //
    kernel = clCreateKernel(program, "matvec", &err);
    if (!kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        return EXIT_FAILURE;
    }
    
    int col=300;
    err=0;
    //attach arguments to the kernal function
    err=clSetKernelArg(kernel,0,sizeof(cl_mem),&a_in);
    err|=clSetKernelArg(kernel,1,sizeof(cl_mem),&b_in);
    err|=clSetKernelArg(kernel,2,sizeof(cl_mem),&c_out);
    err|=clSetKernelArg(kernel,3,sizeof(int),NULL);
    
    if (err)
    {
        printf("Error: Failed to attach arguments to the kernal function!\n");
        return EXIT_FAILURE;
    }
    //write buffers from host to global mem
    err=0;
    err=clEnqueueWriteBuffer(queue, a_in, true, 0, sizeof(double)*size*size, gpua, 0, NULL, NULL);
    err=clEnqueueWriteBuffer(queue, b_in, true, 0, sizeof(double)*size, gpub, 0, NULL, NULL);
    
    if (err)
    {
        printf("Error: Failed to write to buffer!\n");
        return EXIT_FAILURE;
    }
    
    err=0;
    err=clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, NULL);
 
    if (err)
    {
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }
    err=0;
    err= clEnqueueReadBuffer(queue, c_out, true, 0, sizeof(double)*size, gpuc, 0, NULL, NULL);
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
    
  
    
}
    t_tgpu= stopwatch_stop (tgpu);
    fprintf (stderr, "Time to execute gpu mult: %Lg secs\n",t_tgpu);
    
   
    
    return 0;
}

static char * load_program_source(const char *filename)
{
    struct stat statbuf;
    FILE        *fh;
    char        *source;
    
    fh = fopen(filename, "r");
    if (fh == 0)
    {return 0;}
    
    stat(filename, &statbuf);
    source = (char *) malloc(statbuf.st_size + 1);
    fread(source, statbuf.st_size, 1, fh);
    source[statbuf.st_size] = '\0';
    
    return source;
}

