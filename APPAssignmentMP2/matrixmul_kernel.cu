/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"

// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
int TILE_WIDTH=16;
__shared__ float M_s[16][16];
__shared__ float N_s[16][16];
int bx = blockIdx.x; 
int by = blockIdx.y;
int tx = threadIdx.x; int ty = threadIdx.y;
int Row = by * TILE_WIDTH + ty;
int Col = bx * TILE_WIDTH + tx;
float Pvalue = 0;
//if((Row <M.height) && (Col<N.width))
//{

for (int m = 0; m <ceil(M.width/(float)TILE_WIDTH); ++m) {
if(Row<M.height && (m*TILE_WIDTH+tx)<M.width)
M_s[ty][tx] = M.elements[Row*M.width + m*TILE_WIDTH+tx];
else
M_s[ty][tx]=0;
if(Col<N.width && (m*TILE_WIDTH+ty)<N.height)
N_s[ty][tx] = N.elements[(m*TILE_WIDTH+ty)*N.width+Col];
else
N_s[ty][tx]=0;
__syncthreads();
for (int k = 0; k < TILE_WIDTH; ++k)
{
//if(ty<P.height && tx <P.width)
Pvalue += M_s[ty][k] * N_s[k][tx];

}
//Pvalue += M_s[ty][k] * N_s[k][tx];
__syncthreads();
}
if(Row<P.height &&Col<P.width) 	
P.elements[Row*N.width+Col] = Pvalue;
//}
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
