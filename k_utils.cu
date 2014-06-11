#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#pragma once
#define BLOCK_SIZE 32
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
  int width;
  int height;
	double* elements;
} Matrix;

//function to print a matrix
void printMatrix(Matrix A) {
	printf("\n");
	for (int i=0; i<A.height; i++) {
		for (int j=0; j<A.width; j++) {
			printf("%.4f ", A.elements[i*A.width+j]); 
		}
		printf("\n");
	}
	printf("\n");
}

// check if a square matrix is symmetric
__global__
void isSymmetricKernel(Matrix d_A, int *d_result) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= d_A.height || col >= d_A.width) return;
	if(d_A.elements[row*d_A.width+col] != d_A.elements[row + col*d_A.width])
		*(d_result) = 0;
}

int isSymmetric(Matrix A) {
	printf("isSymmetric()\n");
	// load A to device memory
	Matrix d_A;
	d_A.width = A.width;
	d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(double);
	cudaError_t err = cudaMalloc(&d_A.elements, size);
	printf("CUDA malloc A: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
	printf("Copy A to device: %s\n", cudaGetErrorString(err));

	// load result to device memory
	int result = 1;
	int *d_result;
	err = cudaMalloc(&d_result, sizeof(int));
	printf("CUDA malloc d_result: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_result, &result, sizeof(int), cudaMemcpyHostToDevice);	
	printf("Copy result to device: %s\n", cudaGetErrorString(err));

	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (A.width + dimBlock.x - 1)/dimBlock.x, (A.height + dimBlock.y - 1)/dimBlock.y );
	isSymmetricKernel<<<dimGrid, dimBlock>>>(d_A, d_result);
	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	//read result from fdevice memory
	err = cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
	printf("Copy result off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_A.elements);
	cudaFree(d_result);

	return result;
}

//create an m-by-n tiling of a given matrix
__global__
void repmatKernel(Matrix d_A, int m, int n, Matrix d_B) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= d_A.height || col >= d_A.width) return;
	for(int i=0; i < m; i++) {
		for(int j=0; j < n; j++) {
			d_B.elements[(row + i*d_A.height)*d_B.width + (col + j*d_A.width)] = d_A.elements[row*d_A.width + col];
		}
	}
}

//matrix transpose kernel
__global__
void transposeKernel(Matrix d_A, Matrix d_B){
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= d_A.height || col >= d_A.width) return;
	d_B.elements[col*d_B.width + row] = d_A.elements[row*d_A.width + col];
}

// matrix zeros kernel called by zeros()
__global__
void zerosKernel(Matrix d_A) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= d_A.height || col >= d_A.width) return;
	d_A.elements[row*d_A.width+col] = 0;
}

// matrix getRow kernel
__global__
void getRowKernel(Matrix d_In, Matrix d_Out, int num) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= d_In.height || col >= d_In.width) return;
	if(row == num) 
		d_Out.elements[col] = d_In.elements[row*d_In.width+col];
}

// matrix getCol kernel
__global__
void getColKernel(Matrix d_In, Matrix d_Out, int num) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= d_In.height || col >= d_In.width) return;
	if(col == num) 
		d_Out.elements[row] = d_In.elements[row*d_In.width+col];
}

// matSub kernel
__global__
void matSubKernel(Matrix d_A, Matrix d_B, Matrix d_C) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= d_A.height || col >= d_A.width) return;
	d_C.elements[row*d_A.width + col] = d_A.elements[row*d_A.width + col] - d_B.elements[row*d_A.width + col];
}

// exp kernel
__global__ 
void expKernel(Matrix d_D, double sigma) {
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = row * d_D.width + col;
	if(row >= d_D.height || col >= d_D.width) return;
	d_D.elements[idx] = exp(-d_D.elements[idx] * d_D.elements[idx] / sigma);
}

// matAdd kernel
__global__
void matAddKernel(Matrix d_A, Matrix d_B, Matrix d_C) {
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= d_A.height || col >= d_A.width) return;
	d_C.elements[row*d_C.width + col] = d_A.elements[row*d_A.width + col] + d_B.elements[row*d_B.width + col];
}

