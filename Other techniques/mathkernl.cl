#define PADDING         (32)
#define GROUP_DIMX      (32)
#define LOG_GROUP_DIMX  (5)
#define GROUP_DIMY      (2)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void matvec(__global long double *matrix,__global long double *x, __global long double *out ,__local int *count)
{
 //first attempt. one thread per row, not the most sophisticated, but meh.
    
    const uint group_id = get_global_id(0) / get_local_size(0);
    const uint group_size = get_local_size(0);
    
    
    int i,k;
    int leng= 300;
    i=get_global_id(0);
    //j=get_global_id(1);
    
    for(k=0;k<leng;k++)
    {
        out[i]+=matrix[leng*i+k]*x[i];
        
    }
}
