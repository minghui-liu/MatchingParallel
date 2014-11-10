#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <limits.h>

#pragma once
#define BLOCK_SIZE 32
#define BLOCK_SIZE_DIM1 1024

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
  int width;
  int height;
	float* elements;
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

// matrix zeros kernel called by zeros()
__global__
void zerosKernel(Matrix d_A) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= d_A.height || col >= d_A.width) return;
	d_A.elements[row*d_A.width+col] = 0;
}

void zeros(Matrix A) {
	// load A to device memory
	Matrix d_A;
	d_A.width = A.width;
	d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float);
	cudaError_t err = cudaMalloc(&d_A.elements, size);
	printf("CUDA malloc A: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);	
	printf("Copy A to device: %s\n", cudaGetErrorString(err));
	
	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (A.width + dimBlock.x - 1)/dimBlock.x, (A.height + dimBlock.y - 1)/dimBlock.y );
	zerosKernel<<<dimGrid, dimBlock>>>(d_A);
	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read A from device memory
	err = cudaMemcpy(A.elements, d_A.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy A off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_A.elements);
}

//matrix transpose kernel
__global__
void transposeKernel(Matrix d_A, Matrix d_B){
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= d_A.height || col >= d_A.width) return;
	d_B.elements[col*d_B.width + row] = d_A.elements[row*d_A.width + col];
}

void transpose(Matrix In, Matrix Out){
	// load In to device memory
	Matrix d_In;
	d_In.width = In.width;
	d_In.height = In.height;
	size_t size = In.width * In.height * sizeof(float);

	cudaError_t err = cudaMalloc(&d_In.elements, size);
	printf("CUDA malloc In: %s\n", cudaGetErrorString(err));
	cudaMemcpy(d_In.elements, In.elements, size, cudaMemcpyHostToDevice);
	printf("Copy In to device: %s\n", cudaGetErrorString(err));

	// allocate Out on device memory
	Matrix d_Out;
	d_Out.width = Out.width;
	d_Out.height = Out.width;
	size = d_Out.width * d_Out.height * sizeof(float);
	err = cudaMalloc(&d_Out.elements, size);
	printf("CUDA malloc d_Out: %s\n", cudaGetErrorString(err));

	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (In.width + dimBlock.x - 1)/dimBlock.x, (In.height + dimBlock.y - 1)/dimBlock.y );
	transposeKernel<<<dimGrid, dimBlock>>>(d_In, d_Out);
	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read Out from device memory
	err = cudaMemcpy(Out.elements, d_Out.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy d_Out off of device: %s\n",cudaGetErrorString(err));

// free device memory
	cudaFree(d_In.elements);
	cudaFree(d_Out.elements);
}

// matrix ones kernel called by ones()
__global__
void onesKernel(Matrix d_A) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row > d_A.height || col > d_A.width) return;
	d_A.elements[row*d_A.width+col] = 1;
}

void ones(Matrix A) {
	// load A to device memory
	Matrix d_A;
	d_A.width = A.width;
	d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float);
	cudaError_t err = cudaMalloc(&d_A.elements, size);
	printf("CUDA malloc A: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);	
	printf("Copy A to device: %s\n", cudaGetErrorString(err));
	
	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (A.width + dimBlock.x - 1)/dimBlock.x, (A.height + dimBlock.y - 1)/dimBlock.y );
	onesKernel<<<dimGrid, dimBlock>>>(d_A);
	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read A from device memory
	err = cudaMemcpy(A.elements, d_A.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy C off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_A.elements);
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
	size_t size = A.width * A.height * sizeof(float);
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

// check if a matrix is symmetric
__global__
void isSymmetricEpsKernel(Matrix d_A, int *d_result, float eps){

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row > d_A.height || col > d_A.width) return;
	if(d_A.elements[row*d_A.width+col] + eps < d_A.elements[row + col*d_A.width] && 
		d_A.elements[row*d_A.width+col] - eps > d_A.elements[row + col*d_A.width])
	
		*(d_result) = 0;
}

int isSymmetricEps(Matrix A, float eps) {
	// load A to device memory
	Matrix d_A;
	d_A.width = A.width;
	d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float);
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
	isSymmetricEpsKernel<<<dimGrid, dimBlock>>>(d_A, d_result, eps);
	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read result from device memory
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

void repmat(Matrix In, int m, int n, Matrix Out){
	if (Out.height != In.height * m || Out.width != In.width * n) {
		printf("Output matrix has incorrect dimensions!\n");
		return;
	}
	// load In  to device memory
	Matrix d_In;
	d_In.width = In.width;
	d_In.height = In.height;
	size_t size = d_In.width * d_In.height * sizeof(float);
	cudaError_t err = cudaMalloc(&d_In.elements, size);
	printf("CUDA malloc In: %s\n", cudaGetErrorString(err));
	cudaMemcpy(d_In.elements, In.elements, size, cudaMemcpyHostToDevice);
	printf("Copy In to device: %s\n", cudaGetErrorString(err));

	// allocate Out on device memory
	Matrix d_Out;
	d_Out.width = In.width * n;
	d_Out.height = In.height * m;
	size = d_Out.width * d_Out.height * sizeof(float);
	err = cudaMalloc(&d_Out.elements, size);
	printf("CUDA malloc d_Out: %s\n", cudaGetErrorString(err));
	cudaMemcpy(d_Out.elements, Out.elements, size, cudaMemcpyHostToDevice);
	printf("Copy d_Out to device: %s\n", cudaGetErrorString(err));

	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (In.width + dimBlock.x - 1)/dimBlock.x, (In.height + dimBlock.y - 1)/dimBlock.y );
	repmatKernel<<<dimGrid, dimBlock>>>(d_In, m, n, d_Out);
	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read Out from device memory
	err = cudaMemcpy(Out.elements, d_Out.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy d_Out off of device: %s\n",cudaGetErrorString(err));

// free device memory
	cudaFree(d_In.elements);
	cudaFree(d_Out.elements);
}

// matSub kernel
__global__
void matSubKernel(Matrix d_A, Matrix d_B, Matrix d_C) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= d_A.height || col >= d_A.width) return;
	d_C.elements[row*d_A.width + col] = d_A.elements[row*d_A.width + col] - d_B.elements[row*d_A.width + col];
}

void matSub(Matrix A, Matrix B, Matrix C){

	// load A, B to device memory
	Matrix d_A;
	Matrix d_B;
	d_A.width = A.width;
	d_B.width = B.width;
	d_A.height = A.height;
	d_B.height = B.height;
	size_t size = A.width * A.height * sizeof(float);

	cudaError_t err = cudaMalloc(&d_A.elements, size);
	printf("CUDA malloc A: %s\n", cudaGetErrorString(err));
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
	printf("Copy A to device: %s\n", cudaGetErrorString(err));

	err = cudaMalloc(&d_B.elements, size);
	printf("CUDA malloc B: %s\n", cudaGetErrorString(err));
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
	printf("Copy B to device: %s\n", cudaGetErrorString(err));
	
	// allocate C to device memory
	Matrix d_C;
	d_C.width = C.width;
	d_C.height = C.height;
	err = cudaMalloc(&d_C.elements, size);
	printf("CUDA malloc C: %s\n", cudaGetErrorString(err));

	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (A.width + dimBlock.x - 1)/dimBlock.x, (A.height + dimBlock.y - 1)/dimBlock.y );
	matSubKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read C from device memory
	err = cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy C off of device: %s\n", cudaGetErrorString(err));

// free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}


// matAdd kernel
__global__
void matAddKernel(Matrix d_A, Matrix d_B, Matrix d_C) {
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= d_A.height || col >= d_A.width) return;
	d_C.elements[row*d_C.width + col] = d_A.elements[row*d_A.width + col] + d_B.elements[row*d_B.width + col];
}


void matAdd(Matrix A, Matrix B, Matrix C){

	// load A, B to device memory
	Matrix d_A;
	Matrix d_B;
	d_A.width = A.width;
	d_B.width = B.width;
	d_A.height = A.height;
	d_B.height = B.height;
	size_t size = A.width * A.height * sizeof(float);

	cudaError_t err = cudaMalloc(&d_A.elements, size);
	printf("CUDA malloc A: %s\n", cudaGetErrorString(err));
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
	printf("Copy A to device: %s\n", cudaGetErrorString(err));

	err = cudaMalloc(&d_B.elements, size);
	printf("CUDA malloc B: %s\n", cudaGetErrorString(err));
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
	printf("Copy B to device: %s\n", cudaGetErrorString(err));
	
	// allocate C on device memory
	Matrix d_C;
	d_C.width = C.width;
	d_C.height = C.height;
	err = cudaMalloc(&d_C.elements, size);
	printf("CUDA malloc C: %s\n", cudaGetErrorString(err));

	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (A.width + dimBlock.x - 1)/dimBlock.x, (A.height + dimBlock.y - 1)/dimBlock.y );
	matAddKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read C from device memory
	err = cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy C off of device: %s\n", cudaGetErrorString(err));

	// free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}

// matrix matTimesScaler kernel called by matTimesScaler()
__global__
void matTimesScalerKernel(Matrix d_In, float scaler, Matrix d_Out) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= d_In.height || col >= d_In.width) return;
	int idx = row * d_In.width +  col;
	d_Out.elements[idx] = d_In.elements[idx] * scaler;
}


void matTimesScaler(Matrix In, float scaler, Matrix Out) {
	// load In to device memory
	Matrix d_In;
	d_In.width = In.width;
	d_In.height = In.height;
	size_t size = In.width * In.height * sizeof(float);
	cudaError_t err = cudaMalloc(&d_In.elements, size);
	printf("CUDA malloc In: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_In.elements, In.elements, size, cudaMemcpyHostToDevice);	
	printf("Copy input matrix to device: %s\n", cudaGetErrorString(err));
	
	// allocate Out in device memory
	Matrix d_Out;
  d_Out.width = Out.width; d_Out.height = Out.height;
  size = Out.width * Out.height * sizeof(float);
  cudaMalloc(&d_Out.elements, size);

	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (In.width + dimBlock.x - 1)/dimBlock.x, (In.height + dimBlock.y - 1)/dimBlock.y );
	matTimesScalerKernel<<<dimGrid, dimBlock>>>(d_In, scaler, d_Out);
	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read Out from device memory
	err = cudaMemcpy(Out.elements, d_Out.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy output matrix off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_In.elements);
	cudaFree(d_Out.elements);
}

// matrix matPlusScaler kernel called by matPlusScaler()
__global__
void matPlusScalerKernel(Matrix d_In, float scaler, Matrix d_Out) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= d_In.height || col >= d_In.width) return;
	int idx = row * d_In.width +  col;
	d_Out.elements[idx] = d_In.elements[idx] + scaler;
}

void matPlusScaler(Matrix In, float scaler, Matrix Out) {
	// load In to device memory
	Matrix d_In;
	d_In.width = In.width;
	d_In.height = In.height;
	size_t size = In.width * In.height * sizeof(float);
	cudaError_t err = cudaMalloc(&d_In.elements, size);
	printf("CUDA malloc In: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_In.elements, In.elements, size, cudaMemcpyHostToDevice);	
	printf("Copy input matrix to device: %s\n", cudaGetErrorString(err));
	
	// allocate Out in device memory
	Matrix d_Out;
  d_Out.width = Out.width; d_Out.height = Out.height;
  size = Out.width * Out.height * sizeof(float);
  cudaMalloc(&d_Out.elements, size);

	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (In.width + dimBlock.x - 1)/dimBlock.x, (In.height + dimBlock.y - 1)/dimBlock.y );
	matPlusScalerKernel<<<dimGrid, dimBlock>>>(d_In, scaler, d_Out);
	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read Out from device memory
	err = cudaMemcpy(Out.elements, d_Out.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy output matrix off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_In.elements);
	cudaFree(d_Out.elements);

}

// matrix matDiv kernel called by matDiv()
__global__
void matDivKernel(Matrix d_A, Matrix d_B, Matrix d_Out) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = row*d_A.width+col;
	if(row > d_A.height || col > d_A.width) return;
	d_Out.elements[idx] = d_A.elements[idx] / d_B.elements[idx];
}

void matDiv(Matrix A, Matrix B, Matrix Out) {
	if (A.width != B.width || A.height != B.height) {
		printf("Input matrices must have the same dimension!\n");
		return;
	}
	// load A to device memory
	Matrix d_A;
	d_A.width = A.width;
	d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float);
	cudaError_t err = cudaMalloc(&d_A.elements, size);
	printf("CUDA malloc A: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);	
	printf("Copy input matrix A to device: %s\n", cudaGetErrorString(err));
	
	// load B to device memory
	Matrix d_B;
	d_B.width = B.width;
	d_B.height = B.height;
	err = cudaMalloc(&d_B.elements, size);
	printf("CUDA malloc B: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);	
	printf("Copy input matrix B to device: %s\n", cudaGetErrorString(err));
	
	// allocate Out in device memory
	Matrix d_Out;
  d_Out.width = Out.width; d_Out.height = Out.height;
  size = Out.width * Out.height * sizeof(float);
  cudaMalloc(&d_Out.elements, size);

	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (A.width + dimBlock.x - 1)/dimBlock.x, (A.height + dimBlock.y - 1)/dimBlock.y );
	matDivKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_Out);
	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read Out from device memory
	err = cudaMemcpy(Out.elements, d_Out.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy output matrix off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_Out.elements);

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

void getCol(Matrix In, Matrix Out, int num) {
	// load In to device memory
	Matrix d_In;
	d_In.width = In.width;
	d_In.height = In.height;
	size_t size = In.width * In.height * sizeof(float);
	cudaError_t err = cudaMalloc(&d_In.elements, size);
	printf("CUDA malloc In: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_In.elements, In.elements, size, cudaMemcpyHostToDevice);	
	printf("Copy input matrix to device: %s\n", cudaGetErrorString(err));
	
	// allocate Out in device memory
	Matrix d_Out;
  d_Out.width = Out.width; d_Out.height = Out.height;
  size = Out.width * Out.height * sizeof(float);
  cudaMalloc(&d_Out.elements, size);

	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (In.width + dimBlock.x - 1)/dimBlock.x, (In.height + dimBlock.y - 1)/dimBlock.y );
	getColKernel<<<dimGrid, dimBlock>>>(d_In, d_Out, num);
	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read Out from device memory
	err = cudaMemcpy(Out.elements, d_Out.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy output row off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_In.elements);
	cudaFree(d_Out.elements);
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

void getRow(Matrix In, Matrix Out, int num) {
	// load In to device memory
	Matrix d_In;
	d_In.width = In.width;
	d_In.height = In.height;
	size_t size = In.width * In.height * sizeof(float);
	cudaError_t err = cudaMalloc(&d_In.elements, size);
	printf("CUDA malloc In: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_In.elements, In.elements, size, cudaMemcpyHostToDevice);	
	printf("Copy input matrix to device: %s\n", cudaGetErrorString(err));
	
	// allocate Out in device memory
	Matrix d_Out;
  d_Out.width = Out.width; d_Out.height = Out.height;
  size = Out.width * Out.height * sizeof(float);
  cudaMalloc(&d_Out.elements, size);

	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (In.width + dimBlock.x - 1)/dimBlock.x, (In.height + dimBlock.y - 1)/dimBlock.y );
	getRowKernel<<<dimGrid, dimBlock>>>(d_In, d_Out, num);
	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read Out from device memory
	err = cudaMemcpy(Out.elements, d_Out.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy output row off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_In.elements);
	cudaFree(d_Out.elements);
}


// matrix indexOfElement kernel
__global__
void indexOfElementKernel(Matrix d_A, float element, int *index) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row > d_A.height || col > d_A.width) return;
	int idx = row*d_A.width+col;
	if (d_A.elements[idx] == element)
		*(index) = idx;
}

int indexOfElement(Matrix d_A, float element) {
	int index = -1;	

	// allocate d_index on device memory
	int *d_index;
	cudaError_t err = cudaMalloc(&d_index, sizeof(int));
	printf("CUDA malloc index; %s\n", cudaGetErrorString(err));
	cudaMemcpy(d_index, &index, sizeof(int), cudaMemcpyHostToDevice);
	printf("Copy index to device: %s\n", cudaGetErrorString(err));
	
	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (d_A.width + dimBlock.x - 1)/dimBlock.x, (d_A.height + dimBlock.y - 1)/dimBlock.y );
	indexOfElementKernel<<<dimGrid, dimBlock>>>(d_A, element, d_index);
	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read index from device memory
	err = cudaMemcpy(&index, d_index, sizeof(int), cudaMemcpyDeviceToHost);
	printf("Copy index off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_index);
	return index;
}


// matrix reshape kernel called by reshape()
__global__
void reshapeKernel(Matrix d_In, Matrix d_Out) {
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(y > d_In.height || x > d_In.width) return;
	int c = x * d_In.height + y;
	d_Out.elements[(c%d_Out.height)*d_Out.width+(c/d_Out.height)] = d_In.elements[(c%d_In.height)*d_In.width+(c/d_In.height)];
}

void reshape(Matrix In, Matrix Out) {
	// load In to device memory
	Matrix d_In;
	d_In.width = In.width;
	d_In.height = In.height;
	size_t size = In.width * In.height * sizeof(float);
	cudaError_t err = cudaMalloc(&d_In.elements, size);
	printf("CUDA malloc In: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_In.elements, In.elements, size, cudaMemcpyHostToDevice);	
	printf("Copy input matrix to device: %s\n", cudaGetErrorString(err));
	
	// allocate Out in device memory
	Matrix d_Out;
	d_Out.width = Out.width; d_Out.height = Out.height;
	size = Out.width * Out.height * sizeof(float);
	cudaMalloc(&d_Out.elements, size);

	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (In.width + dimBlock.x - 1)/dimBlock.x, (In.height + dimBlock.y - 1)/dimBlock.y );
	reshapeKernel<<<dimGrid, dimBlock>>>(d_In, d_Out);
	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read Out from device memory
	err = cudaMemcpy(Out.elements, d_Out.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy output matrix off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_In.elements);
	cudaFree(d_Out.elements);
}

__global__
void maxReduceKernel(float *elements, int size, float *d_part) {
	int  thread2;
	float temp;
	__shared__ float sdata[BLOCK_SIZE_DIM1];
	
	// Load max from global memory
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		sdata[threadIdx.x] = elements[idx];
	else
		sdata[threadIdx.x] = DBL_MIN;
	
	// Synchronize to make sure data is loaded before starting the comparison
  __syncthreads();

	int nTotalThreads = BLOCK_SIZE_DIM1;
	 
	while(nTotalThreads > 1) {
		int halfPoint = (nTotalThreads >> 1);	// divide by two
		// only the first half of the threads will be active.
	 
		if (threadIdx.x < halfPoint) {
			thread2 = threadIdx.x + halfPoint;
			// Get the shared value stored by another thread 
			temp = sdata[thread2];
			if (temp > sdata[threadIdx.x]) 
				 sdata[threadIdx.x] = temp;
		}
		__syncthreads();
	 
		// Reducing the binary tree size by two:
		nTotalThreads = halfPoint;
	}
	
	// thread 0 copy the max to d_max
	if (threadIdx.x == 0) {
		d_part[blockIdx.x] = sdata[threadIdx.x];
	}
}

float maxOfMatrix(Matrix d_A) {

	// allocate d_part1 on device memory
	float *d_part1;
	cudaError_t err = cudaMalloc(&d_part1, BLOCK_SIZE_DIM1*BLOCK_SIZE_DIM1*sizeof(float));
	printf("CUDA malloc d_part1; %s\n", cudaGetErrorString(err));
	err = cudaMemset(d_part1, DBL_MIN,  BLOCK_SIZE_DIM1*BLOCK_SIZE_DIM1*sizeof(float));
	printf("CUDA memset d_part1 to DBL_MIN: %s\n", cudaGetErrorString(err));	
	
	// allocate d_part2 on device memory
	float *d_part2;
	err = cudaMalloc(&d_part2, BLOCK_SIZE_DIM1*sizeof(float));
	printf("CUDA malloc d_part2; %s\n", cudaGetErrorString(err));
	err = cudaMemset(d_part1, DBL_MIN, BLOCK_SIZE_DIM1*sizeof(float));
	printf("CUDA memset d_part2 to DBL_MIN: %s\n", cudaGetErrorString(err));	
	
	// allocate d_max on device memory
	float *d_max;
	err = cudaMalloc(&d_max, sizeof(float));
	printf("CUDA malloc d_max; %s\n", cudaGetErrorString(err));
	err = cudaMemset(d_max, DBL_MIN, sizeof(float));
	printf("CUDA memset d_max to DBL_MIN: %s\n", cudaGetErrorString(err));

	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE_DIM1);
	dim3 dimGrid((d_A.width*d_A.height + dimBlock.x - 1)/dimBlock.x);
	
	// first pass
	maxReduceKernel<<<dimGrid, dimBlock>>>(d_A.elements, d_A.width*d_A.height, d_part1);
	err = cudaThreadSynchronize();
	printf("Run kernel 1st pass: %s\n", cudaGetErrorString(err));
	
	// second pass
	dimGrid = dim3(BLOCK_SIZE_DIM1);
	maxReduceKernel<<<dimGrid, dimBlock>>>(d_part1, BLOCK_SIZE_DIM1*BLOCK_SIZE_DIM1, d_part2);
	err = cudaThreadSynchronize();
	printf("Run kernel 2nd pass: %s\n", cudaGetErrorString(err));
	
	// third pass
	dimGrid = dim3(1);
	maxReduceKernel<<<dimGrid, dimBlock>>>(d_part2, BLOCK_SIZE_DIM1, d_max);
	err = cudaThreadSynchronize();
	printf("Run kernel 3rd pass: %s\n", cudaGetErrorString(err));

	// read max from device memory
	float max;
	err = cudaMemcpy(&max, d_max, sizeof(float), cudaMemcpyDeviceToHost);
	printf("Copy max off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_part1);
	cudaFree(d_part2);
	cudaFree(d_max);
	
	return max;
}

__global__
void minReduceKernel(float *elements, int size, float *d_part) {
	int  thread2;
	float temp;
	__shared__ float sdata[BLOCK_SIZE_DIM1];
	
	// Load data from global memory
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		sdata[threadIdx.x] = elements[idx];
	else
		sdata[threadIdx.x] = DBL_MAX;
	
	// Synchronize to make sure data is loaded before starting the comparison
  __syncthreads();

	int nTotalThreads = BLOCK_SIZE_DIM1;
	 
	while(nTotalThreads > 1) {
		int halfPoint = (nTotalThreads >> 1);	// divide by two
		// only the first half of the threads will be active.
	 
		if (threadIdx.x < halfPoint) {
			thread2 = threadIdx.x + halfPoint;
			// Get the shared value stored by another thread 
			temp = sdata[thread2];
			if (temp < sdata[threadIdx.x]) 
				 sdata[threadIdx.x] = temp;

		}
		__syncthreads();
	 
		// Reducing the binary tree size by two:
		nTotalThreads = halfPoint;
	}
	
	// thread 0 copy the min to d_min
	if (threadIdx.x == 0) {
		d_part[blockIdx.x] = sdata[threadIdx.x];
	}
}

float minOfMatrix(Matrix d_A) {

	// allocate d_part1 on device memory
	float *d_part1;
	cudaError_t err = cudaMalloc(&d_part1, BLOCK_SIZE_DIM1*BLOCK_SIZE_DIM1*sizeof(float));
	printf("CUDA malloc d_part1; %s\n", cudaGetErrorString(err));
	err = cudaMemset(d_part1, DBL_MAX,  BLOCK_SIZE_DIM1*BLOCK_SIZE_DIM1*sizeof(float));
	printf("CUDA memset d_part1 to DBL_MAX: %s\n", cudaGetErrorString(err));	
	
	// allocate d_part2 on device memory
	float *d_part2;
	err = cudaMalloc(&d_part2, BLOCK_SIZE_DIM1*sizeof(float));
	printf("CUDA malloc d_part2; %s\n", cudaGetErrorString(err));
	err = cudaMemset(d_part1, DBL_MAX, BLOCK_SIZE_DIM1*sizeof(float));
	printf("CUDA memset d_part2 to DBL_MAX: %s\n", cudaGetErrorString(err));	
	
	// allocate d_min on device memory
	float *d_min;
	err = cudaMalloc(&d_min, sizeof(float));
	printf("CUDA malloc d_min; %s\n", cudaGetErrorString(err));
	err = cudaMemset(d_min, DBL_MAX, sizeof(float));
	printf("CUDA memset d_min to DBL_MAX: %s\n", cudaGetErrorString(err));

	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE_DIM1);
	dim3 dimGrid((d_A.width*d_A.height + dimBlock.x - 1)/dimBlock.x);
	
	// first pass
	minReduceKernel<<<dimGrid, dimBlock>>>(d_A.elements, d_A.width*d_A.height, d_part1);
	err = cudaThreadSynchronize();
	printf("Run kernel 1st pass: %s\n", cudaGetErrorString(err));
	
	// second pass
	dimGrid = dim3(BLOCK_SIZE_DIM1);
	minReduceKernel<<<dimGrid, dimBlock>>>(d_part1, BLOCK_SIZE_DIM1*BLOCK_SIZE_DIM1, d_part2);
	err = cudaThreadSynchronize();
	printf("Run kernel 2nd pass: %s\n", cudaGetErrorString(err));
	
	// third pass
	dimGrid = dim3(1);
	minReduceKernel<<<dimGrid, dimBlock>>>(d_part2, BLOCK_SIZE_DIM1, d_min);
	err = cudaThreadSynchronize();
	printf("Run kernel 3rd pass: %s\n", cudaGetErrorString(err));

	// read max from device memory
	float min;
	err = cudaMemcpy(&min, d_min, sizeof(float), cudaMemcpyDeviceToHost);
	printf("Copy min off of device: %s\n",cudaGetErrorString(err));
	
	// free device memory
	cudaFree(d_part1);
	cudaFree(d_part2);
	cudaFree(d_min);
	
	return min;
}


__global__
void sumReduceKernel(float *elements, int size, float *d_part) {
	int  thread2;
	__shared__ float sdata[BLOCK_SIZE_DIM1];
	
	// Load elements from global memory
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		sdata[threadIdx.x] = elements[idx];
	else
		sdata[threadIdx.x] = 0;
	
	// Synchronize to make sure data is loaded before starting the comparison
  __syncthreads();

	int nTotalThreads = BLOCK_SIZE_DIM1;
	 
	while(nTotalThreads > 1) {
		int halfPoint = (nTotalThreads >> 1);	// divide by two
		// only the first half of the threads will be active.
	 
		if (threadIdx.x < halfPoint) {
			thread2 = threadIdx.x + halfPoint;
			// Get the shared value stored by another thread and sum it to sdata
			sdata[threadIdx.x] += sdata[thread2];

		}
		__syncthreads();
	 
		// Reducing the binary tree size by two:
		nTotalThreads = halfPoint;
	}
	
	// thread 0 copy the max to d_max
	if (threadIdx.x == 0) {
		d_part[blockIdx.x] = sdata[threadIdx.x];
	}
}

float matSum(Matrix d_A) {

	// allocate d_part1 on device memory
	float *d_part1;
	cudaError_t err = cudaMalloc(&d_part1, BLOCK_SIZE_DIM1*BLOCK_SIZE_DIM1*sizeof(float));
	printf("CUDA malloc d_part1; %s\n", cudaGetErrorString(err));
	err = cudaMemset(d_part1, 0,  BLOCK_SIZE_DIM1*BLOCK_SIZE_DIM1*sizeof(float));
	printf("CUDA memset d_part1 to 0: %s\n", cudaGetErrorString(err));	
	
	// allocate d_part2 on device memory
	float *d_part2;
	err = cudaMalloc(&d_part2, BLOCK_SIZE_DIM1*sizeof(float));
	printf("CUDA malloc d_part2; %s\n", cudaGetErrorString(err));
	err = cudaMemset(d_part1, 0, BLOCK_SIZE_DIM1*sizeof(float));
	printf("CUDA memset d_part2 to 0: %s\n", cudaGetErrorString(err));	
	
	// allocate d_sum on device memory
	float *d_sum;
	err = cudaMalloc(&d_sum, sizeof(float));
	printf("CUDA malloc d_sum; %s\n", cudaGetErrorString(err));
	err = cudaMemset(d_sum, 0, sizeof(float));
	printf("CUDA memset d_sum to 0: %s\n", cudaGetErrorString(err));

	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE_DIM1);
	dim3 dimGrid((d_A.width*d_A.height + dimBlock.x - 1)/dimBlock.x);
	
	// first pass
	sumReduceKernel<<<dimGrid, dimBlock>>>(d_A.elements, d_A.width*d_A.height, d_part1);
	err = cudaThreadSynchronize();
	printf("Run kernel 1st pass: %s\n", cudaGetErrorString(err));
	
	// second pass
	dimGrid = dim3(BLOCK_SIZE_DIM1);
	sumReduceKernel<<<dimGrid, dimBlock>>>(d_part1, BLOCK_SIZE_DIM1*BLOCK_SIZE_DIM1, d_part2);
	err = cudaThreadSynchronize();
	printf("Run kernel 2nd pass: %s\n", cudaGetErrorString(err));
	
	// third pass
	dimGrid = dim3(1);
	sumReduceKernel<<<dimGrid, dimBlock>>>(d_part2, BLOCK_SIZE_DIM1, d_sum);
	err = cudaThreadSynchronize();
	printf("Run kernel 3rd pass: %s\n", cudaGetErrorString(err));

	// read sum from device memory
	float sum;
	err = cudaMemcpy(&sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
	printf("Copy sum off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_part1);
	cudaFree(d_part2);
	cudaFree(d_sum);
	
	return sum;
}


__global__
void maxOfMatrixRow(Matrix d_A, Matrix d_col) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	float max = d_A.elements[row*d_A.width];
	for (int col=0; col<d_A.width; col++) {
		max = (d_A.elements[row*d_A.width+col] > max)? d_A.elements[row*d_A.width+col] : max;
	}
	d_col.elements[row] = max;
}


__global__
void maxOfMatrixCol(Matrix d_A, Matrix d_row) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float max = d_A.elements[col];
	for (int row=0; row<d_A.height; row++) {
		max = (d_A.elements[row*d_A.width+col] > max)? d_A.elements[row*d_A.width+col] : max;
	}
	d_row.elements[col] = max;
}

__global__
void sumOfMatrixRowKernel(Matrix d_In, Matrix d_sumCol) {
	int idx =  blockIdx.x * blockDim.x + threadIdx.x;	
	dim3 dimBlock(BLOCK_SIZE_DIM1);
	dim3 dimGrid( (d_In.width + dimBlock.x - 1)/dimBlock.x );
	
	// two pass sum reduction
	// allocate d_part
	float *d_part = (float*)malloc(dimGrid.x * sizeof(float));
	// allocate d_sum 
	float *d_sum = (float*)malloc(sizeof(float));
	memset(d_sum, 0, sizeof(float));

	// first pass
	sumReduceKernel<<<dimGrid, dimBlock>>>(d_In.elements, d_In.width, d_part);
	
	// second pass
	dimGrid = dim3(1);
	sumReduceKernel<<<dimGrid, dimBlock>>>(d_part, dimGrid.x, d_sum);

	// write d_sum to d_sumCol
	d_sumCol.elements[idx] = *d_sum;

	// free device memory
	free(d_part);
}

void sumOfMatrixRow(Matrix In, Matrix sumCol) {
	// load In to device memory
	Matrix d_In;
	d_In.width = In.width;
	d_In.height = In.height;
	size_t size = In.width * In.height * sizeof(float);
	cudaError_t err = cudaMalloc(&d_In.elements, size);
	printf("CUDA malloc In: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_In.elements, In.elements, size, cudaMemcpyHostToDevice);	
	printf("Copy input matrix to device: %s\n", cudaGetErrorString(err));
	
	// allocate sumCol in device memory
	Matrix d_sumCol;
	d_sumCol.width = sumCol.width; d_sumCol.height = sumCol.height;
	size = sumCol.width * sumCol.height * sizeof(float);
	err = cudaMalloc(&d_sumCol.elements, size);
	printf("CUDA malloc sumCol: %s\n", cudaGetErrorString(err));
	
	// lauch one thread for each row to do the sum
	dim3 dimBlock(BLOCK_SIZE_DIM1);
	dim3 dimGrid( (In.height + dimBlock.x - 1)/dimBlock.x );
	sumOfMatrixRowKernel<<<dimGrid, dimBlock>>>(d_In, d_sumCol);
	err = cudaThreadSynchronize();
	printf("Run sum of matrix kernel: %s\n", cudaGetErrorString(err));
	
	// read sumCol from device memory
	err = cudaMemcpy(sumCol.elements, d_sumCol.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy sumCol off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_In.elements);
	cudaFree(d_sumCol.elements);
}

__global__
void colSumReduceKernel(Matrix d_In, int colNum, float *d_part) {
	int  thread2;
	__shared__ float sdata[BLOCK_SIZE_DIM1];
	
	// Load elements from global memory
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < d_In.height)
		sdata[threadIdx.x] = d_In.elements[idx*d_In.width + colNum];
	else
		sdata[threadIdx.x] = 0;
	
	// Synchronize to make sure data is loaded before starting the comparison
  __syncthreads();

	int nTotalThreads = BLOCK_SIZE_DIM1;
	 
	while(nTotalThreads > 1) {
		int halfPoint = (nTotalThreads >> 1);	// divide by two
		// only the first half of the threads will be active.
	 
		if (threadIdx.x < halfPoint) {
			thread2 = threadIdx.x + halfPoint;
			// Get the shared value stored by another thread and sum it to sdata
			sdata[threadIdx.x] += sdata[thread2];

		}
		__syncthreads();
	 
		// Reducing the binary tree size by two:
		nTotalThreads = halfPoint;
	}
	
	// thread 0 copy the max to d_max
	if (threadIdx.x == 0) {
		d_part[blockIdx.x] = sdata[threadIdx.x];
	}
}

__global__
void sumOfMatrixColKernelOne(Matrix d_In, Matrix d_sumRow) {
	int idx =  blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= d_In.width) return;
	
	dim3 dimBlock(BLOCK_SIZE_DIM1);
	dim3 dimGrid( (d_In.height + dimBlock.x - 1)/dimBlock.x );
	
	// two pass sum reduction
	// allocate d_part
	float *d_part = (float*)malloc(dimGrid.x * sizeof(float));
	memset(d_part, 0, dimGrid.x * sizeof(float));
	// allocate d_sum 
	float *d_sum = (float*)malloc(sizeof(float));
	memset(d_sum, 0, sizeof(float));

	// first pass
	colSumReduceKernel<<<dimGrid, dimBlock>>>(d_In, idx, d_part);
	
	// second pass
	dimGrid = dim3(1);
	sumReduceKernel<<<dimGrid, dimBlock>>>(d_part, dimGrid.x, d_sum);

	// write d_sum to d_sumCol
	d_sumRow.elements[idx] = *d_sum;

	// free device memory
	free(d_part);
}

void sumOfMatrixCol(Matrix In, Matrix sumRow) {
	// load In to device memory
	Matrix d_In;
	d_In.width = In.width;
	d_In.height = In.height;
	size_t size = In.width * In.height * sizeof(float);
	cudaError_t err = cudaMalloc(&d_In.elements, size);
	printf("CUDA malloc In: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_In.elements, In.elements, size, cudaMemcpyHostToDevice);	
	printf("Copy input matrix to device: %s\n", cudaGetErrorString(err));
	
	// allocate sumRow in device memory
	Matrix d_sumRow;
	d_sumRow.width = sumRow.width; d_sumRow.height = sumRow.height;
	size = sumRow.width * sumRow.height * sizeof(float);
	err = cudaMalloc(&d_sumRow.elements, size);
	printf("CUDA malloc sumrow: %s\n", cudaGetErrorString(err));
	
	// lauch one thread for each col to do the sum
	dim3 dimBlock(BLOCK_SIZE_DIM1);
	dim3 dimGrid( (In.width + dimBlock.x - 1)/dimBlock.x );
	sumOfMatrixRowKernel<<<dimGrid, dimBlock>>>(d_In, d_sumRow);
	err = cudaThreadSynchronize();
	printf("Run sum of matrix row kernel: %s\n", cudaGetErrorString(err));
	
	// read sumRow from device memory
	err = cudaMemcpy(sumRow.elements, d_sumRow.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy sumRow off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_In.elements);
	cudaFree(d_sumRow.elements);
}
