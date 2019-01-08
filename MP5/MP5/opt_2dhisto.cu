#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"
#include "ref_2dhisto.h"
#include "opt_2dhisto.h"

/*void opt_2dhisto( uint32_t *input_data,int inputheight,int inputwidth ,uint32_t *input_bins)
{
    /* This function should only contain grid setup 
       code and a call to the GPU histogramming kernel. 
       Any memory allocations and transfers must be done 
       outside this function 
int totalsize=inputheight*inputwidth;
opt_2dhisto_kernel<<<ceil(totalsize/ 1024), 1024>>>(input_data,inputheight,inputwidth, input_bins);
}*/
__global__ void opt_2dhisto_kernel(uint32_t *input_data,int inputheight,int inputwidth,uint32_t *input_bins)
{
int size=inputheight*inputwidth;
    __shared__ unsigned int private_histo[HISTO_WIDTH];
   if (threadIdx.x < HISTO_WIDTH) private_histo[threadIdx.x] = 0;
   __syncthreads();
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (i < size) {
         atomicAdd( &(private_histo[input_data[i]]), 1);
         i += stride;
    }
__syncthreads();

  if (threadIdx.x < HISTO_WIDTH) 
     atomicAdd( &input_bins[threadIdx.x],private_histo[threadIdx.x] );


}


void opt_2dhisto( uint32_t *input_data,int inputheight,int inputwidth ,uint32_t *input_bins)
{
    /* This function should only contain grid setup
       code and a call to the GPU histogramming kernel.
       Any memory allocations and transfers must be done
       outside this function */
int totalsize=20*2048;
  cudaMemset(input_bins, 0, sizeof(uint32_t) * HISTO_WIDTH);
opt_2dhisto_kernel<<<ceil(totalsize/ BLOCK_SIZE), BLOCK_SIZE>>>(input_data,inputheight,inputwidth, input_bins);
  cudaDeviceSynchronize(); 
}


/* Include below the implementation of any other functions you need */
uint32_t * AllocateDataInDevice(int height,int width,int size)
{
//printf("HI");
uint32_t * data;
cudaMalloc((void**)&data,height*width*size);
return data;
}
void FreeDeviceData(uint32_t * data)
{
cudaFree(data);
}
void CopyFromHostToDevice(uint32_t *device_data,uint32_t **host_data,int inputrow,int inputcol,int size)
{
int totalsize=size*inputcol;
for(int i=0;i<inputrow;i++)
{
cudaMemcpy(device_data,host_data[i],totalsize,cudaMemcpyHostToDevice);
device_data+=inputcol;
//host_data+=inputcol;
}
}
void CopyFromDeviceToHost(uint32_t *host,uint32_t *device,int size,int elementsize)
{
cudaMemcpy(host,device,elementsize*size,cudaMemcpyDeviceToHost);
for(int i = 0; i < HISTO_WIDTH * HISTO_HEIGHT; i++)
        if(host[i] > 255){
			host[i] = 255;
		}
		
}

