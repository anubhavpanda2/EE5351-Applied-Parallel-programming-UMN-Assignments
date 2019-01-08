#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define TILE_SIZE 1024
// You can use any other block size you wish.
#define BLOCK_SIZE 128
#define DBLOCK_SIZE BLOCK_SIZE*2

// Host Helper Functions (allocate your own data structure...)



// Device Functions



// Kernel Functions


__global__ void blockscan(unsigned int *blocksum,unsigned int *outArray,unsigned int *inArray,int numElements)
{

__shared__ unsigned int  scan_array[DBLOCK_SIZE];
//unsigned int lastElement;
unsigned int t = threadIdx.x;
unsigned int bin=blockIdx.x;
unsigned int start = DBLOCK_SIZE*bin;
if(start+t<numElements)
scan_array[t] = inArray[start + t];
else
scan_array[t]=0;
 if (start+ BLOCK_SIZE + t < numElements)
       scan_array[BLOCK_SIZE + t] = inArray[start + BLOCK_SIZE + t];
    else
       scan_array[BLOCK_SIZE + t] = 0;
__syncthreads();
//if(t==0) 
  //     lastElement = scan_array[DBLOCK_SIZE-1];
//__syncthreads();
//prescan
int stride = 1;
 while(stride <= BLOCK_SIZE)
{
    int index = (t+1)*stride*2 - 1;
        if(index < DBLOCK_SIZE)
	        scan_array[index] += scan_array[index-stride];
		    stride = stride*2;
		        __syncthreads();
}

if(t==0)
blocksum[bin]=scan_array[DBLOCK_SIZE-1];


if (t==0)
{ 
scan_array[DBLOCK_SIZE-1] = 0;
}
 stride = BLOCK_SIZE; 

while(stride > 0) 
{   int index = (t+1)*stride*2 - 1;
  if(index < DBLOCK_SIZE) 
  {      float temp = scan_array[index];
  scan_array[index] += scan_array[index-stride];
  scan_array[index-stride] = temp;  
  } 
  stride = stride / 2;
  __syncthreads();
} 
if(start+t<numElements)
outArray[start+t]=scan_array[t];
else
outArray[start+t]=0;
if(start + BLOCK_SIZE + t < numElements){
		outArray[start + BLOCK_SIZE + t] = scan_array[t + BLOCK_SIZE];
	}
	else{
		outArray[start + BLOCK_SIZE + t] = 0;
	}
//if(t==0)
//blocksum[bin]=scan_array[DBLOCK_SIZE-1]+lastElement;

}
__global__ void totalsum(unsigned int *outArray,unsigned int *blocksum,int numElements)
{
__shared__ unsigned int add;
int index= blockIdx.x * DBLOCK_SIZE+threadIdx.x;
if(threadIdx.x==0)
add=blocksum[blockIdx.x];
__syncthreads();
if(index<numElements)
{
outArray[index]+=add;
outArray[index+BLOCK_SIZE]+=add;
}
}
void blockscanrecursion(unsigned int *outArray, int numElements)
{
unsigned int *blocksum;
int size=ceil(numElements/((float)DBLOCK_SIZE));
cudaMalloc( (void**) &blocksum, sizeof(unsigned int) * (size));
 blockscan<<<size, BLOCK_SIZE>>>(blocksum, outArray, outArray, numElements);
    if(size > 1)
    {
        blockscanrecursion(blocksum, size);
        totalsum<<<size , BLOCK_SIZE>>>(outArray,blocksum,numElements);
    }
	cudaFree(blocksum);

}
// **===-------- Modify the body of this function -----------===**
// You may need to make multiple kernel calls. Make your own kernel
// functions in this file, and then call them from here.
// Note that the code has been modified to ensure numElements is a multiple 
// of TILE_SIZE
void prescanArray(unsigned int *outArray, unsigned int *inArray, int numElements)
{
dim3 dimblock(BLOCK_SIZE);
//dim3 dimgrid(ceil(numElements/((float)BLOCK_SIZE*2.0)));
unsigned int *blocksum;
int size=ceil(numElements/((float)DBLOCK_SIZE));
//printf("%d",numElements);
dim3 dimgrid(size);
cudaMalloc((void**)&blocksum, size);
blockscan<<<dimgrid,dimblock>>>(blocksum,outArray,inArray,numElements);
if(size>1)
{
blockscanrecursion(blocksum,size);
totalsum<<<dimgrid,dimblock>>>(outArray,blocksum,numElements);
}
cudaFree(blocksum);
}
// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_
