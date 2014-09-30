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

/*** matlib functions listed in alphabetical order *****/


/************************ G ************************/
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
	//printf("getCol()\n");
	// load In to device memory
	Matrix d_In;
	d_In.width = In.width;
	d_In.height = In.height;
	size_t size = In.width * In.height * sizeof(double);
	cudaError_t err = cudaMalloc(&d_In.elements, size);
	//printf("CUDA malloc In: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_In.elements, In.elements, size, cudaMemcpyHostToDevice);	
	//printf("Copy input matrix to device: %s\n", cudaGetErrorString(err));
	
	// allocate Out in device memory
	Matrix d_Out;
  d_Out.width = Out.width; d_Out.height = Out.height;
  size = Out.width * Out.height * sizeof(double);
  err = cudaMalloc(&d_Out.elements, size);
  //printf("CUDA malloc Out: %s\n", cudaGetErrorString(err));	

	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (In.width + dimBlock.x - 1)/dimBlock.x, (In.height + dimBlock.y - 1)/dimBlock.y );
	getColKernel<<<dimGrid, dimBlock>>>(d_In, d_Out, num);
	err = cudaThreadSynchronize();
	//printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read Out from device memory
	err = cudaMemcpy(Out.elements, d_Out.elements, size, cudaMemcpyDeviceToHost);
	//printf("Copy output matrix off of device: %s\n",cudaGetErrorString(err));

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

/************************ I ************************/

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
	//printf("isSymmetric()\n");
	// load A to device memory
	Matrix d_A;
	d_A.width = A.width;
	d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(double);
	cudaError_t err = cudaMalloc(&d_A.elements, size);
	//printf("CUDA malloc A: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
	//printf("Copy A to device: %s\n", cudaGetErrorString(err));

	// load result to device memory
	int result = 1;
	int *d_result;
	err = cudaMalloc(&d_result, sizeof(int));
	//printf("CUDA malloc d_result: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_result, &result, sizeof(int), cudaMemcpyHostToDevice);	
	//printf("Copy result to device: %s\n", cudaGetErrorString(err));

	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (A.width + dimBlock.x - 1)/dimBlock.x, (A.height + dimBlock.y - 1)/dimBlock.y );
	isSymmetricKernel<<<dimGrid, dimBlock>>>(d_A, d_result);
	err = cudaThreadSynchronize();
	//printf("Run kernel: %s\n", cudaGetErrorString(err));

	//read result from fdevice memory
	err = cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
	//printf("Copy result off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_A.elements);
	cudaFree(d_result);

	return result;
}

/************************ M ************************/

// matAdd kernel
__global__
void matAddKernel(Matrix d_A, Matrix d_B, Matrix d_C) {
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= d_A.height || col >= d_A.width) return;
	d_C.elements[row*d_C.width + col] = d_A.elements[row*d_A.width + col] + d_B.elements[row*d_B.width + col];
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
	//printf("matDiv()\n");
	if (A.width != B.width || A.height != B.height) {
		printf("Input matrices must have the same dimension!\n");
		return;
	}
	// load A to device memory
	Matrix d_A;
	d_A.width = A.width;
	d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(double);
	cudaError_t err = cudaMalloc(&d_A.elements, size);
	//printf("CUDA malloc A: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);	
	//printf("Copy input matrix A to device: %s\n", cudaGetErrorString(err));
	
	// load B to device memory
	Matrix d_B;
	d_B.width = B.width;
	d_B.height = B.height;
	err = cudaMalloc(&d_B.elements, size);
	//printf("CUDA malloc B: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);	
	//printf("Copy input matrix B to device: %s\n", cudaGetErrorString(err));
	
	// allocate Out in device memory
	Matrix d_Out;
  d_Out.width = Out.width; d_Out.height = Out.height;
  size = Out.width * Out.height * sizeof(double);
  cudaMalloc(&d_Out.elements, size);

	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (A.width + dimBlock.x - 1)/dimBlock.x, (A.height + dimBlock.y - 1)/dimBlock.y );
	matDivKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_Out);
	err = cudaThreadSynchronize();
	//printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read Out from device memory
	err = cudaMemcpy(Out.elements, d_Out.elements, size, cudaMemcpyDeviceToHost);
	//printf("Copy output matrix off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_Out.elements);

}

// matrix matPlusScaler kernel called by matPlusScaler()
__global__
void matPlusScalerKernel(Matrix d_In, double scaler, Matrix d_Out) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= d_In.height || col >= d_In.width) return;
	int idx = row * d_In.width +  col;
	d_Out.elements[idx] = d_In.elements[idx] + scaler;
}

void matPlusScaler(Matrix In, double scaler, Matrix Out) {
	//printf("matPlusScaler()\n");
	// load In to device memory
	Matrix d_In;
	d_In.width = In.width;
	d_In.height = In.height;
	size_t size = In.width * In.height * sizeof(double);
	cudaError_t err = cudaMalloc(&d_In.elements, size);
	//printf("CUDA malloc In: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_In.elements, In.elements, size, cudaMemcpyHostToDevice);	
	//printf("Copy input matrix to device: %s\n", cudaGetErrorString(err));
	
	// allocate Out in device memory
	Matrix d_Out;
  d_Out.width = Out.width; d_Out.height = Out.height;
  size = Out.width * Out.height * sizeof(double);
  cudaMalloc(&d_Out.elements, size);

	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (In.width + dimBlock.x - 1)/dimBlock.x, (In.height + dimBlock.y - 1)/dimBlock.y );
	matPlusScalerKernel<<<dimGrid, dimBlock>>>(d_In, scaler, d_Out);
	err = cudaThreadSynchronize();
	//printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read Out from device memory
	err = cudaMemcpy(Out.elements, d_Out.elements, size, cudaMemcpyDeviceToHost);
	//printf("Copy output matrix off of device: %s\n",cudaGetErrorString(err));

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
	//printf("matSub()\n");
	// load A, B to device memory
	Matrix d_A;
	Matrix d_B;
	d_A.width = A.width;
	d_B.width = B.width;
	d_A.height = A.height;
	d_B.height = B.height;
	size_t size = A.width * A.height * sizeof(double);

	cudaError_t err = cudaMalloc(&d_A.elements, size);
	//printf("CUDA malloc A: %s\n", cudaGetErrorString(err));
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
	//printf("Copy A to device: %s\n", cudaGetErrorString(err));

	err = cudaMalloc(&d_B.elements, size);
	//printf("CUDA malloc B: %s\n", cudaGetErrorString(err));
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
	//printf("Copy B to device: %s\n", cudaGetErrorString(err));
	
	// allocate C to device memory
	Matrix d_C;
	d_C.width = C.width;
	d_C.height = C.height;
	err = cudaMalloc(&d_C.elements, size);
	//printf("CUDA malloc C: %s\n", cudaGetErrorString(err));

	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (A.width + dimBlock.x - 1)/dimBlock.x, (A.height + dimBlock.y - 1)/dimBlock.y );
	matSubKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
	err = cudaThreadSynchronize();
	//printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read C from device memory
	err = cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
	//printf("Copy C off of device: %s\n", cudaGetErrorString(err));

// free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}

// matrix matTimesScaler kernel called by matTimesScaler()
__global__
void matTimesScalerKernel(Matrix d_In, double scaler, Matrix d_Out) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= d_In.height || col >= d_In.width) return;
	int idx = row * d_In.width +  col;
	d_Out.elements[idx] = d_In.elements[idx] * scaler;
}

void matTimesScaler(Matrix In, double scaler, Matrix Out) {
	//printf("matTimesScaler()\n");
	// load In to device memory
	Matrix d_In;
	d_In.width = In.width;
	d_In.height = In.height;
	size_t size = In.width * In.height * sizeof(double);
	cudaError_t err = cudaMalloc(&d_In.elements, size);
	//printf("CUDA malloc In: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_In.elements, In.elements, size, cudaMemcpyHostToDevice);	
	//printf("Copy input matrix to device: %s\n", cudaGetErrorString(err));
	
	// allocate Out in device memory
	Matrix d_Out;
  d_Out.width = Out.width; d_Out.height = Out.height;
  size = Out.width * Out.height * sizeof(double);
  cudaMalloc(&d_Out.elements, size);

	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (In.width + dimBlock.x - 1)/dimBlock.x, (In.height + dimBlock.y - 1)/dimBlock.y );
	matTimesScalerKernel<<<dimGrid, dimBlock>>>(d_In, scaler, d_Out);
	err = cudaThreadSynchronize();
	//printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read Out from device memory
	err = cudaMemcpy(Out.elements, d_Out.elements, size, cudaMemcpyDeviceToHost);
	//printf("Copy output matrix off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_In.elements);
	cudaFree(d_Out.elements);
}

// need a better parallelized version
__global__
void maxOfMatrixRow(Matrix d_A, Matrix d_col) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	double max = d_A.elements[row*d_A.width];
	for (int col=0; col<d_A.width; col++) {
		max = (d_A.elements[row*d_A.width+col] > max)? d_A.elements[row*d_A.width+col] : max;
	}
	d_col.elements[row] = max;
}

/************************ O ************************/
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
	size_t size = A.width * A.height * sizeof(double);
	cudaError_t err = cudaMalloc(&d_A.elements, size);
	//printf("CUDA malloc A: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);	
	//printf("Copy A to device: %s\n", cudaGetErrorString(err));
	
	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (A.width + dimBlock.x - 1)/dimBlock.x, (A.height + dimBlock.y - 1)/dimBlock.y );
	onesKernel<<<dimGrid, dimBlock>>>(d_A);
	err = cudaThreadSynchronize();
	//printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read A from device memory
	err = cudaMemcpy(A.elements, d_A.elements, size, cudaMemcpyDeviceToHost);
	//printf("Copy C off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_A.elements);
}


/************************ R ************************/

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

// matrix reshape kernel called by reshape()
__global__
void reshapeKernel(Matrix d_In, Matrix d_Out) {
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(y >= d_In.height || x >= d_In.width) return;
	int c = x * d_In.height + y;
	d_Out.elements[(c%d_Out.height)*d_Out.width+(c/d_Out.height)] = d_In.elements[(c%d_In.height)*d_In.width+(c/d_In.height)];

}

void reshape(Matrix In, Matrix Out) {
	//printf("reshape()\n");
	// load In to device memory
	Matrix d_In;
	d_In.width = In.width;
	d_In.height = In.height;
	size_t size = In.width * In.height * sizeof(double);
	cudaError_t err = cudaMalloc(&d_In.elements, size);
	//printf("CUDA malloc In: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_In.elements, In.elements, size, cudaMemcpyHostToDevice);	
	//printf("Copy input matrix to device: %s\n", cudaGetErrorString(err));
	
	// allocate Out in device memory
	Matrix d_Out;
	d_Out.width = Out.width; d_Out.height = Out.height;
	size = Out.width * Out.height * sizeof(double);
	cudaMalloc(&d_Out.elements, size);

	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (In.width + dimBlock.x - 1)/dimBlock.x, (In.height + dimBlock.y - 1)/dimBlock.y );
	reshapeKernel<<<dimGrid, dimBlock>>>(d_In, d_Out);
	err = cudaThreadSynchronize();
	//printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read Out from device memory
	err = cudaMemcpy(Out.elements, d_Out.elements, size, cudaMemcpyDeviceToHost);
	//printf("Copy output matrix off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_In.elements);
	cudaFree(d_Out.elements);
}

/************************ S ************************/
__global__
void sumOfMatrixColKernel(Matrix d_A, Matrix d_row) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col >= d_A.width) return;
	for (int row=0; row<d_A.height; row++) {
		d_row.elements[col] += d_A.elements[row*d_A.width+col];
	}
}

void sumOfMatrixCol(Matrix In, Matrix Out) {
	// load In to device memory
	Matrix d_In;
	d_In.width = In.width;
	d_In.height = In.height;
	size_t size = In.width * In.height * sizeof(double);
	cudaError_t err = cudaMalloc(&d_In.elements, size);
	//printf("CUDA malloc In: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_In.elements, In.elements, size, cudaMemcpyHostToDevice);	
	//printf("Copy input matrix to device: %s\n", cudaGetErrorString(err));
	
	// allocate Out in device memory
	Matrix d_Out;
	d_Out.width = Out.width; d_Out.height = Out.height;
	size = Out.width * Out.height * sizeof(double);
	cudaMalloc(&d_Out.elements, size);

	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE_DIM1);
	dim3 dimGrid( (In.width + dimBlock.x - 1)/dimBlock.x );
	sumOfMatrixColKernel<<<dimGrid, dimBlock>>>(d_In, d_Out);
	err = cudaThreadSynchronize();
	//printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read Out from device memory
	err = cudaMemcpy(Out.elements, d_Out.elements, size, cudaMemcpyDeviceToHost);
	//printf("Copy output matrix off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_In.elements);
	cudaFree(d_Out.elements);
}

/************************ T ************************/

//matrix transpose kernel
__global__
void transposeKernel(Matrix d_A, Matrix d_B){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= d_A.height || col >= d_A.width) return;
	d_B.elements[col*d_B.width+row] = d_A.elements[row*d_A.width+col];
}

void transpose(Matrix In, Matrix Out) {
	//printf("transpose()\n");
	// load In to device memory
	Matrix d_In;
	d_In.width = In.width;
	d_In.height = In.height;
	size_t size = In.width * In.height * sizeof(double);

	cudaError_t err = cudaMalloc(&d_In.elements, size);
	//printf("CUDA malloc In: %s\n", cudaGetErrorString(err));
	cudaMemcpy(d_In.elements, In.elements, size, cudaMemcpyHostToDevice);
	//printf("Copy In to device: %s\n", cudaGetErrorString(err));

	// allocate Out on device memory
	Matrix d_Out;
	d_Out.width = Out.width;
	d_Out.height = Out.height;
	size = d_Out.width * d_Out.height * sizeof(double);
	err = cudaMalloc(&d_Out.elements, size);
	//printf("CUDA malloc d_Out: %s\n", cudaGetErrorString(err));

	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (In.width + dimBlock.x - 1)/dimBlock.x, (In.height + dimBlock.y - 1)/dimBlock.y );
	transposeKernel<<<dimGrid, dimBlock>>>(d_In, d_Out);
	err = cudaThreadSynchronize();
	//printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read Out from device memory
	err = cudaMemcpy(Out.elements, d_Out.elements, size, cudaMemcpyDeviceToHost);
	//printf("Copy d_Out off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_In.elements);
	cudaFree(d_Out.elements);
}

/************************ Z ************************/

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
	size_t size = A.width * A.height * sizeof(double);
	cudaError_t err = cudaMalloc(&d_A.elements, size);
	//printf("CUDA malloc A: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);	
	//printf("Copy A to device: %s\n", cudaGetErrorString(err));
	
	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (A.width + dimBlock.x - 1)/dimBlock.x, (A.height + dimBlock.y - 1)/dimBlock.y );
	zerosKernel<<<dimGrid, dimBlock>>>(d_A);
	err = cudaThreadSynchronize();
	//printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read A from device memory
	err = cudaMemcpy(A.elements, d_A.elements, size, cudaMemcpyDeviceToHost);
	//printf("Copy A off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_A.elements);
}

