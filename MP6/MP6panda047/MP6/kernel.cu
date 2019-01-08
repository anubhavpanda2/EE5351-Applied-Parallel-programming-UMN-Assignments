#include <stdio.h>
#define BLOCK_SIZE 500
__global__ void spmv_csr_kernel(unsigned int dim, unsigned int *csrRowPtr, 
    unsigned int *csrColIdx, float *csrData, float *inVector, 
    float *outVector) {
   int row=blockDim.x*blockIdx.x+threadIdx.x;
   if(row<dim)
   {
   float res=0;
   int row_st=csrRowPtr[row];
   int row_end=csrRowPtr[row+1];
   for(int j=row_st;j<row_end;j++)
   {
   res+=csrData[j]*inVector[csrColIdx[j]];

   }
   outVector[row]=res;
   }

}

__global__ void spmv_jds_kernel(unsigned int dim, unsigned int *jdsRowPerm, 
    unsigned int *jdsRowNNZ, unsigned int *jdsColStartIdx, 
    unsigned int *jdsColIdx, float *jdsData, float* inVector,
    float *outVector) {

    int row=blockDim.x*blockIdx.x+threadIdx.x;
    if(row<dim)
    {
    float res=0;
    for(int j=0;j<jdsRowNNZ[row];j++)
    {
   int  idx=row+jdsColStartIdx[j];
   res+=jdsData[idx]*inVector[jdsColIdx[idx]];
    }
    outVector[jdsRowPerm[row]]=res;


    }

}

void spmv_csr(unsigned int dim, unsigned int *csrRowPtr, unsigned int *csrColIdx, 
    float *csrData, float *inVector, float *outVector) {
spmv_csr_kernel <<<ceil(dim/(float)BLOCK_SIZE),BLOCK_SIZE>>>(dim, csrRowPtr, csrColIdx, csrData, inVector, outVector);
}

void spmv_jds(unsigned int dim, unsigned int *jdsRowPerm, unsigned int *jdsRowNNZ, 
    unsigned int *jdsColStartIdx, unsigned int *jdsColIdx, float *jdsData, 
    float* inVector, float *outVector) {

spmv_jds_kernel<<<ceil(dim/(float)BLOCK_SIZE),BLOCK_SIZE, BLOCK_SIZE>>>(dim, jdsRowPerm, jdsRowNNZ, jdsColStartIdx, jdsColIdx, jdsData, inVector, outVector);
}














