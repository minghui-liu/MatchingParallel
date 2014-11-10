#include <stdio.h>
#include "nearestDSmax_RE.cu"
#include "matlib.cu"

#define BLOCK_SIZE 32
#define BLOCK_SIZE_DIM1 1024

__global__
void negInfRow(Matrix d_In, int row) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col >= d_In.width) return;
	d_In.elements[row*d_In.width+col] = -INFINITY;
}

__global__
void negInfCol(Matrix d_In, int col) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= d_In.height) return;
	d_In.elements[row*d_In.width+col] = -INFINITY;
}

void soft2hard(Matrix soft, int numberOfMatches, Matrix hard) {
	// allocate d_soft on device memory
	Matrix d_soft;
	d_soft.height = soft.height;
	d_soft.width = soft.width;
	size_t size = d_soft.height * d_soft.width * sizeof(float);
	cudaError_t err = cudaMalloc(&d_soft.elements, size);
	//printf("CUDA malloc d_soft: %s\n", cudaGetErrorString(err));
	err = cudaMemcpy(d_soft.elements, soft.elements, size, cudaMemcpyHostToDevice);	
	//printf("Copy soft to device: %s\n", cudaGetErrorString(err));

	// allocate d_hard on device memory
	Matrix d_hard;
	d_hard.height = hard.height;
	d_hard.width = hard.width;
	size = d_hard.height * d_hard.width * sizeof(float);
	err = cudaMalloc(&d_hard.elements, size);
	//printf("CUDA malloc d_hard: %s\n", cudaGetErrorString(err));	
		
	// make d_hard an all zero matrix, invoke zeros kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (d_hard.width+dimBlock.x-1)/dimBlock.x, (d_hard.height+dimBlock.y-1)/dimBlock.y );
	zerosKernel<<<dimGrid, dimBlock>>>(d_hard);
	
	// allocate maxSoft one device
	Matrix d_maxSoft;
	d_maxSoft.height = d_soft.height;
	d_maxSoft.width = 1;
	err = cudaMalloc(&d_maxSoft.elements, d_maxSoft.height*sizeof(float));
	//printf("CUDA malloc d_maxSoft: %s\n", cudaGetErrorString(err));	
	
	// allocate d_soft_r on device
	Matrix d_soft_r;
	d_soft_r.height = 1;
	d_soft_r.width = d_soft.width;
	err = cudaMalloc(&d_soft_r.elements, d_soft_r.width*sizeof(float));
	//printf("CUDA malloc d_soft_r: %s\n", cudaGetErrorString(err));	
	
	for (int i=0; i < numberOfMatches; i++) {	

		// maxSoft = max(soft,[],2);
		// invoke maxOfMatrixRow kernel
		//printf("maxOfMatrixRow()\n");
		dimBlock = dim3(BLOCK_SIZE_DIM1);
		dimGrid = dim3( (d_maxSoft.height + dimBlock.x - 1)/dimBlock.x );
		maxOfMatrixRow<<<dimGrid, dimBlock>>>(d_soft, d_maxSoft);
		err = cudaThreadSynchronize();
		//printf("Run maxOfMatrixRow kernel: %s\n", cudaGetErrorString(err));
		
		printf("before dummy and r\n");		
		// [dummy,r] = max(maxSoft);
		thrust::device_vector<float> D_maxSoft(d_maxSoft.elements, d_maxSoft.elements + d_maxSoft.width * d_maxSoft.height);
		thrust::detail::normal_iterator<thrust::device_ptr<float> > dummyIt = thrust::max_element(D_maxSoft.begin(), D_maxSoft.end());
		int r = dummyIt - D_maxSoft.begin();
		printf("after dummy and r\n");		

		// soft(r,:)
		// invoke getRow kernel
		dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);		
		dimGrid = dim3( (d_soft.width + dimBlock.x - 1)/dimBlock.x ,(d_soft.height + dimBlock.y - 1)/dimBlock.y );
		getRowKernel<<<dimGrid, dimBlock>>>(d_soft, d_soft_r, r);
	
		printf("before val and c\n");	
		// [val,c] = max(soft(r,:));
		thrust::device_vector<float> D_soft_r(d_soft_r.elements, d_soft_r.elements + d_soft_r.width * d_soft_r.height);
		thrust::detail::normal_iterator<thrust::device_ptr<float> > valIt = thrust::max_element(D_soft_r.begin(), D_soft_r.end());
		float val = *valIt;
		int c = valIt - D_soft_r.begin();
		printf("after val and c\n");		

		if (val < 0) { 
			return;
		}
		
		// hard(r,c) = 1
		*(hard.elements + r * hard.width + c) = 1;
		// soft(r,:) = -Inf;
		dimBlock = dim3(BLOCK_SIZE_DIM1);
		dimGrid = dim3((d_soft.width + dimBlock.x - 1)/dimBlock.x);
		negInfRow<<<dimGrid, dimBlock>>>(d_soft, r);
		// soft(:,c) = -Inf;
		dimGrid = dim3((d_soft.height + dimBlock.x - 1)/dimBlock.x);
		negInfCol<<<dimGrid, dimBlock>>>(d_soft, c);
	}	
	
	// free device memory
	cudaFree(d_soft.elements);
	cudaFree(d_maxSoft.elements);
	cudaFree(d_soft_r.elements);
	
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
void hypergraphMatching(Matrix Y, int numberOfMatches, Matrix X, Matrix Z) {

	Matrix maxRowSum, maxColSum;
	maxRowSum.height = Y.height;
	maxRowSum.width = 1;
	maxRowSum.elements = (float*)malloc(maxRowSum.height*sizeof(float));
	maxColSum.height = 1;
	maxColSum.width = Y.width;
	maxColSum.elements = (float*)malloc(maxColSum.width*sizeof(float));
	
	ones(maxRowSum);
	ones(maxColSum);
	
	nearestDSmax_RE(Y, maxRowSum, maxColSum, numberOfMatches, 1000, 0.01, Z);
	soft2hard(Z, numberOfMatches, X);
}
