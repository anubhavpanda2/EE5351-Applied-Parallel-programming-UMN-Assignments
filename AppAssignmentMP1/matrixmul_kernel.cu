/* Matrix multiplication: P = M * N.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"

// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
  //Multiply the two matrices
int Row=blockIdx.y*blockDim.y+threadIdx.y;
int Col=blockIdx.x*blockDim.x+threadIdx.x;
if((Row <HM) && (Col<WN))
	{
		float p=0.0;
		for(int i=0;i<HN;i++)
			p+=M.elements[Row*WM+i]*N.elements[i*HM+Col];
		P.elements[WP*Row+Col]=p;
	}
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
