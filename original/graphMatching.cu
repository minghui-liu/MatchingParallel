#include <stdio.h>
#include <math.h>
#include "hypergraphMatching.cu"
#include "matlib.cu"

#define BLOCK_SIZE 32

// exp kernel
__global__
void expKernel(Matrix d_D, double sigma) {
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= d_D.height || col >= d_D.width) return;
	int idx = row * d_D.width + col;
	d_D.elements[idx] = exp(-d_D.elements[idx] * d_D.elements[idx] / sigma);
}

__global__
void marginalize(Matrix d_G1, Matrix d_G2t, double sigma, Matrix d_Y) {
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	// if out of boundary return
	if (y >= d_Y.height || x >= d_Y.width) return;

	// create d_G1_col and d_G2_row	
	Matrix d_G1_col, d_G2t_row;
	d_G1_col.height = d_G1.height;
	d_G1_col.width = 1;
	d_G1_col.elements = (double*)malloc(d_G1_col.width * d_G1_col.height * sizeof(double));
	d_G2t_row.height = 1;
	d_G2t_row.width = d_G2t.width;
	d_G2t_row.elements = (double*)malloc(d_G2t_row.width * d_G2t_row.height * sizeof(double));
	
	// create d_D, d_D1, d_D2
	Matrix d_D, d_D1, d_D2;
	d_D.height = d_D1.height = d_D2.height = d_Y.height;
	d_D.width = d_D1.width = d_D2.width = d_Y.width;	
	d_D.elements = (double*)malloc(d_D.width * d_D.height * sizeof(double));
	d_D1.elements = (double*)malloc(d_D1.width * d_D1.height * sizeof(double));
	d_D2.elements = (double*)malloc(d_D2.width * d_D2.height * sizeof(double));
	
	// calculate D
	
	// G1(:,i) invoke getCol kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (d_G1.width + dimBlock.x - 1)/dimBlock.x, (d_G1.height + dimBlock.y - 1)/dimBlock.y );
	getColKernel<<<dimGrid, dimBlock>>>(d_G1, d_G1_col, y);
	
	// G2t(j,:) invoke getRow kernel
	dimGrid = dim3( (d_G2t.width + dimBlock.x - 1)/dimBlock.x, (d_G2t.height + dimBlock.y - 1)/dimBlock.y );
	getRowKernel<<<dimGrid, dimBlock>>>(d_G2t, d_G2t_row, x);
	
	// repmat(G1(:,i),1,n2) invoke repmat kernel
	dimGrid = dim3( (d_G1_col.width + dimBlock.x - 1)/dimBlock.x, (d_G1_col.height + dimBlock.y - 1)/dimBlock.y );
	repmatKernel<<<dimGrid, dimBlock>>>(d_G1_col, 1, d_G2t.width, d_D1);

	// repmat(G2t(j,:),n1,1) invoke repmat kernel
	dimGrid = dim3( (d_G2t_row.width + dimBlock.x - 1)/dimBlock.x, (d_G2t_row.height + dimBlock.y - 1)/dimBlock.y );
	repmatKernel<<<dimGrid, dimBlock>>>(d_G2t_row, d_G1.height, 1, d_D2);
	
	// d_D1 - d_D2 invoke matSub kernel
	dimGrid = dim3( (d_D.width + dimBlock.x - 1)/dimBlock.x, (d_D.height + dimBlock.y - 1)/dimBlock.y );
	matSubKernel<<<dimGrid, dimBlock>>>(d_D1, d_D2, d_D);
		
	// exp((-d.*d)./sigma) invoke exp kernel
	expKernel<<<dimGrid, dimBlock>>>(d_D, sigma);
	
	// write to Y
	matAddKernel<<<dimGrid, dimBlock>>>(d_Y, d_D, d_Y);
	//matAtomicAddKernel<<<dimGrid,dimBlock>>>(d_Y, d_D);

	// free memory space
	free(d_G1_col.elements);
	free(d_G2t_row.elements);
	free(d_D.elements);
	free(d_D1.elements);
	free(d_D2.elements);
}

void graphMatching(Matrix G1, Matrix G2, double sigma, int numberOfMatches, Matrix X, Matrix Z, Matrix Y) {
	/****************************************************************************************	
	Algorithm due to R. Zass and A. Shashua.,
 	'Probabilistic Graph and Hypergraph Matching.',
 	Computer Vision and Pattern Recognition (CVPR) Anchorage, Alaska, June 2008.

 	G1  				An size1 by size1 symmetric matrix, with the weight of the first graph edges.
 	G2  				An size2 by size2 symmetric matrix, with the weight of the second graph edges.
 	sigma 	 			Kernel parameter for edge-to-edge correlations.
 	numberOfMatches  	number of matches required. 

 	X [Output]  	a size1 by size2 matrix with the hard matching results.
             		The i,j entry is one iff the i-th feature of the first object
             		match the j-th feature of the second object. Zero otherwise.
 
	Z [Output]  	a size1 by size2 matrix with the soft matching results.
             		The i,j entry is the probablity that the i-th feature of the
             		first object match the j-th feature of the second object.
 
	Y [Output]  	Debug information.
	*****************************************************************************************/
	if (isSymmetric(G1) == 0)
		printf("G1 is not symmetric!\n");
	if (isSymmetric(G2) == 0)
		printf("G2 is not symmetric!\n");
	
	// load G1 to device memory
	Matrix d_G1;
	d_G1.width = G1.width;
	d_G1.height = G1.height;
	size_t size = d_G1.width * d_G1.height * sizeof(double);
	cudaError_t err = cudaMalloc(&d_G1.elements, size);
	//printf("CUDA malloc d_G1: %s\n", cudaGetErrorString(err));	
	err = cudaMemcpy(d_G1.elements, G1.elements, size, cudaMemcpyHostToDevice);	
	//printf("Copy G1 to device: %s\n", cudaGetErrorString(err));
	
	// load G2 to device memory
	Matrix d_G2;
	d_G2.width = G2.height;
	d_G2.height = G2.width;
	size = d_G2.width * d_G2.height * sizeof(double);
	err = cudaMalloc(&d_G2.elements, size);
	//printf("CUDA malloc d_G2: %s\n", cudaGetErrorString(err));	
	err = cudaMemcpy(d_G2.elements, G2.elements, size, cudaMemcpyHostToDevice);	
	//printf("Copy G2 to device: %s\n", cudaGetErrorString(err));
	
	// transpose G2	
	// allocate G2t on device memory
	Matrix d_G2t;
	d_G2t.width = G2.height;
	d_G2t.height = G2.width;
	size = d_G2t.width * d_G2t.height * sizeof(double);
	err = cudaMalloc(&d_G2t.elements, size);
	//printf("CUDA malloc G2t: %s\n", cudaGetErrorString(err));	
	
	// invoke transpose kernel
	//printf("transpose(G2)\n");
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (d_G2.width + dimBlock.x - 1)/dimBlock.x, (d_G2.height + dimBlock.y - 1)/dimBlock.y );
	transposeKernel<<<dimGrid, dimBlock>>>(d_G2, d_G2t);
	err = cudaThreadSynchronize();
	//printf("Run transpose kernel: %s\n", cudaGetErrorString(err));
	
	// free d_G2
	cudaFree(d_G2.elements);
	
	// make Y an all zero matrix
	// allocate Y on device memory
	Matrix d_Y;
	d_Y.height = Y.height;
	d_Y.width = Y.width; 
	size = d_Y.width * d_Y.height * sizeof(double);
	err = cudaMalloc(&d_Y.elements, size);
	//printf("CUDA malloc d_Y: %s\n", cudaGetErrorString(err));
	// invoke zeros kernel
	dimGrid = dim3( (d_Y.width+dimBlock.x-1)/dimBlock.x, (d_Y.height+dimBlock.y-1)/dimBlock.y );
	zerosKernel<<<dimGrid, dimBlock>>>(d_Y);

	marginalize<<<dimGrid, dimBlock>>>(d_G1, d_G2t, sigma, d_Y);
	err = cudaThreadSynchronize();
	//printf("Run marginalize kernel: %s\n", cudaGetErrorString(err));

	// read Y from device memory
	size = Y.width * Y.height * sizeof(double);
	err = cudaMemcpy(Y.elements, d_Y.elements, size, cudaMemcpyDeviceToHost);
	//printf("Copy Y off of device: %s\n",cudaGetErrorString(err));
		
	// free some device memory
	cudaFree(d_G1.elements);
	cudaFree(d_G2t.elements);
	cudaFree(d_Y.elements);

	// call hypergraphMatching()
	hypergraphMatching(Y, numberOfMatches, X, Z);
}

