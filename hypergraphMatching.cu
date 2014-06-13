#include <stdio.h>
#include "k_utils.cu"
#include "nearestDSmax_RE.cu"

#define BLOCK_SIZE 32

__global__
void negInfRow(Matrix d_In, int num) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= d_In.height || col >= d_In.width) return;
	if(row == num) 
		d_In.elements[row*d_In.width+col] = -INFINITY;
}

__global__
void negInfCol(Matrix d_In, int num) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= d_In.height || col >= d_In.width) return;
	if(col == num) 
		d_In.elements[row*d_In.width+col] = -INFINITY;
}

void soft2hard(Matrix d_soft_original, int numberOfMatches, Matrix d_hard) {
	// make a copy of soft
	Matrix d_soft;
	d_soft.height = d_soft_original.height;
	d_soft.width = d_soft_original.width;
	size_t size = d_soft.height * d_soft.width * sizeof(double);
	cudaError_t err = cudaMemcpy(d_d_soft.elements, d_soft_original.elements, size, cudaMemcpyDeviceToDevice);	
	printf("Making a copy of soft: %s\n", cudaGetErrorString(err));

	// make d_hard an all zero matrix, invoke zeros kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (d_hard.width+dimBlock.x-1)/dimBlock.x, (d_hard.height+dimBlock.y-1)/dimBlock.y );
	zerosKernel<<<dimGrid, dimBlock>>>(d_hard);
	
	// allocate maxSoft one device
	Matrix d_maxSoft;
	d_maxSoft.height = d_soft.height;
	d_maxSoft.width = 1;
	err = cudaMalloc(&d_maxSoft.elements, d_maxSoft.height * sizeof(double));
	printf("CUDA malloc d_maxSoft: %s\n", cudaGetErrorString(err));	
		
	for (int i=0; i < numberOfMatches; i++) {	
		// maxSoft = max(soft,[],2);
		// invoke maxOfMatrixRow kernel
		printf("maxOfMatrixRow()\n");
		dimBlock = dim3(BLOCK_SIZE);
		dimGrid = dim3( (d_maxSoft.height + dimBlock.x - 1)/dimBlock.x );
		maxOfMatrixRow<<<dimGrid, dimBlock>>>(d_soft, d_maxSoft);
		err = cudaThreadSynchronize();
		printf("Run maxOfMatrixRow kernel: %s\n", cudaGetErrorString(err));
		
		// [dummy,r] = max(maxSoft);
		dummy = maxOfMatrix(d_maxSoft);
		r = indexOfElement(d_maxSoft, dummy);
		
		// soft(r,:) invoke getRow kernel
		Matrix d_soft_r;
		d_soft_r.height = 1;
		d_soft_r.width = d_soft.width;		
		dimGrid.x = dim3( (d_soft.width + dimBlock.x - 1)/dimBlock.x ,(d_soft.height + dimBlock.y - 1)/dimBlock.y );
		getRowKernel<<<dimGrid, dimBlock>>>(d_soft, d_soft_r, r);
		// [val,c] = max(soft(r,:));
		val = maxOfMatrix(d_soft_r);
		c = indexOfElement(d_soft_r, val);
		
		if (val < 0) { 
			return;
		}
		
		// hard(r,c) = 1
		err = cudaMemset(d_hard+r*d_hard.width+c, 1, sizeof(double));
		printf("CUDA memset d_hard(r,c) to 1: %s\n", cudaGetErrorString(err));
		
		// soft(r,:) = -Inf;
		negInfRow(d_soft, r);
		// soft(:,c) = -Inf;
		negInfCol(d_soft, c);
	}
	
}


/*******************************************************************************
 function [X,Z] = hypergraphMatching (Y, numberOfMatches)

 Optimal soft hyergraph matching.

 Algorithm due to R. Zass and A. Shashua.,
 'Probabilistic Graph and Hypergraph Matching.',
 Computer Vision and Pattern Recognition (CVPR) Anchorage, Alaska, June 2008.

 Y - Marginalization of the hyperedge-to-hyperedge correspondences matrix.
 numberOfMatches - number of matches required.

 X [Output] - an n1 by n2 matrix with the hard matching results.
             The i,j entry is one iff the i-th feature of the first object
             match the j-th feature of the second object. Zero otherwise.
 Z [Output] - an n1 by n2 matrix with the soft matching results.
             The i,j entry is the probablity that the i-th feature of the
             first object match the j-th feature of the second object.

 See also:
 - graphMatching() as an example on how to use this for graphs with a
  specific similarity function.

 Author: Ron Zass, zass@cs.huji.ac.il, www.cs.huji.ac.il/~zass
*******************************************************************************/
void hypergraphMatching(Matrix d_Y, int numberOfMatches, Matrix d_X, Matrix d_Z) {
	Matrix d_maxRowSum, d_maxColSum;
	d_maxRowSum.height = d_Y.height;
	d_maxRowSum.width = 1;
	d_maxColSum.height = 1;
	d_maxColSum.width = d_Y.width;
	size_t size = d_maxRowSum.width * d_maxRowSum.height * sizeof(double);
	cudaError_t err = cudaMalloc(&d_maxRowSum.elements, size);
	printf("CUDA malloc d_maxRowSum: %s\n", cudaGetErrorString(err));
	size = d_maxColSum.width * d_maxColSum.height * sizeof(double);
	err = cudaMalloc(&d_maxColSum.elements, size);
	printf("CUDA malloc d_maxColSum: %s\n", cudaGetErrorString(err));
	
	nearestDSmax_RE(d_Y, d_maxRowSum, d_maxColSum, numberOfMatches, 1000, 0.01, d_Z);
	soft2hard(Z, numberOfMatches, X);
	
	// free device memory
	cudaFree(d_maxRowSum.elements);
	cudaFree(d_maxColSum.elements);
}
