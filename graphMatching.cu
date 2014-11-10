#include <stdio.h>
#include <math.h>
#include "hypergraphMatching.cu"
#include "matlib.cu"

#define BLOCK_SIZE 32

__global__
void marginalize(Matrix, Matrix, float, Matrix); 

void graphMatching(Matrix G1, Matrix G2, float sigma, int numberOfMatches, Matrix X, Matrix Z, Matrix Y) {
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
	size_t size = d_G1.width * d_G1.height * sizeof(float);
	cudaError_t err = cudaMalloc(&d_G1.elements, size);
	printf("CUDA malloc d_G1: %s\n", cudaGetErrorString(err));	
	err = cudaMemcpy(d_G1.elements, G1.elements, size, cudaMemcpyHostToDevice);	
	printf("Copy G1 to device: %s\n", cudaGetErrorString(err));
	
	// load G2 to device memory
	Matrix d_G2;
	d_G2.width = G2.height;
	d_G2.height = G2.width;
	size = d_G2.width * d_G2.height * sizeof(float);
	err = cudaMalloc(&d_G2.elements, size);
	printf("CUDA malloc d_G2: %s\n", cudaGetErrorString(err));	
	err = cudaMemcpy(d_G2.elements, G2.elements, size, cudaMemcpyHostToDevice);	
	printf("Copy G2 to device: %s\n", cudaGetErrorString(err));
	
	// transpose G2	
	// allocate G2t on device memory
	Matrix d_G2t;
	d_G2t.width = G2.height;
	d_G2t.height = G2.width;
	size = d_G2t.width * d_G2t.height * sizeof(float);
	err = cudaMalloc(&d_G2t.elements, size);
	printf("CUDA malloc G2t: %s\n", cudaGetErrorString(err));	

	// invoke transpose kernel
	//printf("transpose(G2)\n");
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (d_G2.width + dimBlock.x - 1)/dimBlock.x, (d_G2.height + dimBlock.y - 1)/dimBlock.y );
	transposeKernel<<<dimGrid, dimBlock>>>(d_G2, d_G2t);
	err = cudaThreadSynchronize();
	printf("Run transpose kernel: %s\n", cudaGetErrorString(err));
	
	// free d_G2
	cudaFree(d_G2.elements);
	
	// make Y an all zero matrix
	// allocate Y on device memory
	Matrix d_Y;
	d_Y.height = Y.height;
	d_Y.width = Y.width; 
	size = d_Y.width * d_Y.height * sizeof(float);
	err = cudaMalloc(&d_Y.elements, size);
	printf("CUDA malloc d_Y: %s\n", cudaGetErrorString(err));
	// invoke zeros kernel
	dimGrid = dim3( (d_Y.width+dimBlock.x-1)/dimBlock.x, (d_Y.height+dimBlock.y-1)/dimBlock.y );
	zerosKernel<<<dimGrid, dimBlock>>>(d_Y);

	marginalize<<<dimGrid, dimBlock>>>(d_G1, d_G2t, sigma, d_Y);
	err = cudaThreadSynchronize();
	//printf("Run marginalize kernel: %s\n", cudaGetErrorString(err));

	// read Y from device memory
	size = Y.width * Y.height * sizeof(float);
	err = cudaMemcpy(Y.elements, d_Y.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy Y off of device: %s\n",cudaGetErrorString(err));
	
	// free some device memory
	cudaFree(d_G1.elements);
	cudaFree(d_G2t.elements);
	cudaFree(d_Y.elements);

	// call hypergraphMatching()
	hypergraphMatching(Y, numberOfMatches, X, Z);
}

__global__
void marginalize(Matrix d_G1, Matrix d_G2t, float sigma, Matrix d_Y) {
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	// if out of boundary return
	if (i >= d_Y.height || j >= d_Y.width) return;

	for (int r=0; r < d_Y.height; r++) {
		for (int c=0; c < d_Y.width; c++) {
			float d = d_G1.elements[r * d_G1.width + i] - d_G2t.elements[j * d_G2t.width + c];
			float e = exp(-d * d / sigma);
			//d_Y.elements[r * d_Y.width + c] += e;
			atomicAdd(d_Y.elements + r * d_Y.width + c, e);
		}
	}
	
}


