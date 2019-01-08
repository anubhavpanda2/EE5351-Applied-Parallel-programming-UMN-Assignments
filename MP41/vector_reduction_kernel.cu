#ifndef _SCAN_NAIVE_KERNEL_H_
#define _SCAN_NAIVE_KERNEL_H_

// **===--------------------- Modify this function -----------------------===**
//! @param g_data  input data in global memory
//                  result is expected in index 0 of g_data
//! @param n        input number of elements to reduce from input data
// **===------------------------------------------------------------------===**
__global__ void reduction(float *g_data, int n)
{
__shared__ float partialSum[2*16];
unsigned int t = threadIdx.x;
unsigned int start = 2*blockDim.x*blockIdx.x;
partialSum[t] = g_data[start + t];
partialSum[blockDim.x + t] = 
	g_data[start + blockDim.x + t];
for (unsigned int stride = blockDim.x; 
	stride >= 1;  stride >>= 1) 
{
  __syncthreads();
  if (t < stride)
    partialSum[t] += partialSum[t+stride];
}
if(t==0)
{
g_data[blockIdx.x]=partialSum[t];
//printf("hello");
}
}

#endif // #ifndef _SCAN_NAIVE_KERNEL_H_
