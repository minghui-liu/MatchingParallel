#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#define BLOCK_SIZE 16

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

// matrix zeros kernel called by zeros()
__global__
void zerosKernel(Matrix d_A) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row > d_A.height || col > d_A.width) return;
	d_A.elements[row*d_A.width+col] = 0;
}

void zeros(Matrix A) {
	// load A to device memory
	Matrix d_A;
	d_A.width = A.width;
	d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(double);
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
	printf("Copy C off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_A.elements);
}

//matrix transpose kernel called by transpose()
__global__
void transposeKernel(Matrix d_A, Matrix d_B){
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= d_A.height || col >= d_A.width) return;
	d_B.elements[col*d_B.width + row] = d_A.elements[row*d_A.width + col];
}

void transpose(Matrix A, Matrix B){

// load A and B to device memory
	Matrix d_A;
	Matrix d_B;
	d_A.width = A.width;
	d_B.width = B.width;
	d_A.height = A.height;
	d_B.height = B.height;
	size_t size = A.width * A.height * sizeof(double);

	cudaError_t err = cudaMalloc(&d_A.elements, size);
	printf("CUDA malloc A: %s\n", cudaGetErrorString(err));
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
	printf("Copy A to device: %s\n", cudaGetErrorString(err));

	cudaError_t errB = cudaMalloc(&d_B.elements, size);
	printf("CUDA malloc B: %s\n", cudaGetErrorString(errB));
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
	printf("Copy B to device: %s\n", cudaGetErrorString(errB));

// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (A.width + dimBlock.x - 1)/dimBlock.x, (A.height + dimBlock.y - 1)/dimBlock.y );
	transposeKernel<<<dimGrid, dimBlock>>>(d_A, d_B);
	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

// read A from device memory
	err = cudaMemcpy(A.elements, d_A.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy C off of device: %s\n",cudaGetErrorString(err));
	errB = cudaMemcpy(B.elements, d_B.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy off of device: %s\n", cudaGetErrorString(errB));

// free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
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
	size_t size = A.width * A.height * sizeof(double);
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
	printf("Copy C off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_A.elements);
}

// check if a matrix is symmetric
__global__
void isSymmetricKernel(Matrix d_A, Matrix d_B){

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row > d_A.height || col > d_A.width) return;
	if(d_A.elements[row*d_A.width+col] == d_A.elements[row + col*d_A.width])
		return;
	else
		d_B.elements[row*d_B.width + col] = 1;
}

void isSymmetric(Matrix A, Matrix B) {
// load A to device memory
	Matrix d_A;
	d_A.width = A.width;
	d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(double);
	cudaError_t errA = cudaMalloc(&d_A.elements, size);
	printf("CUDA malloc A: %s\n", cudaGetErrorString(errA));	
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
	printf("Copy A to device: %s\n", cudaGetErrorString(errA));

// load B to device memory
	Matrix d_B;
	d_B.width = B.width;
	d_B.height = B.height;
	cudaError_t errB = cudaMalloc(&d_B.elements, size);
	printf("CUDA malloc A: %s\n", cudaGetErrorString(errB));	
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);	
	printf("Copy A to device: %s\n", cudaGetErrorString(errB));

// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (A.width + dimBlock.x - 1)/dimBlock.x, (A.height + dimBlock.y - 1)/dimBlock.y );
	isSymmetricKernel<<<dimGrid, dimBlock>>>(d_A, d_B);
	cudaError_t err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

// read A from device memory
	errA = cudaMemcpy(A.elements, d_A.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy A off of device: %s\n",cudaGetErrorString(errA));

//read B from device memory
	errB = cudaMemcpy(B.elements, d_B.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy A off of device: %s\n",cudaGetErrorString(errB));

// free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
}

// check if a matrix is symmetric
__global__
void isSymmetricEpsKernel(Matrix d_A, Matrix d_B, double eps){

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row > d_A.height || col > d_A.width) return;
	if(d_A.elements[row*d_A.width+col] + eps >= d_A.elements[row + col*d_A.width] || 
		d_A.elements[row*d_A.width+col] - eps <= d_A.elements[row + col*d_A.width])
		return;
	else
		d_B.elements[row*d_B.width + col] = 1;
}

void isSymmetricEps(Matrix A, Matrix B, double eps) {
// load A to device memory
	Matrix d_A;
	d_A.width = A.width;
	d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(double);
	cudaError_t errA = cudaMalloc(&d_A.elements, size);
	printf("CUDA malloc A: %s\n", cudaGetErrorString(errA));	
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
	printf("Copy A to device: %s\n", cudaGetErrorString(errA));

// load B to device memory
	Matrix d_B;
	d_B.width = B.width;
	d_B.height = B.height;
	cudaError_t errB = cudaMalloc(&d_B.elements, size);
	printf("CUDA malloc A: %s\n", cudaGetErrorString(errB));	
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);	
	printf("Copy A to device: %s\n", cudaGetErrorString(errB));

// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (A.width + dimBlock.x - 1)/dimBlock.x, (A.height + dimBlock.y - 1)/dimBlock.y );
	isSymmetricEpsKernel<<<dimGrid, dimBlock>>>(d_A, d_B, eps);
	cudaError_t err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

// read A from device memory
	errA = cudaMemcpy(A.elements, d_A.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy A off of device: %s\n",cudaGetErrorString(errA));

//read B from device memory
	errB = cudaMemcpy(B.elements, d_B.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy A off of device: %s\n",cudaGetErrorString(errB));

// free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
}

//create an m-by-n tiling of a given matrix
__global__
void repmatKernel(Matrix d_A, Matrix d_B){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= d_A.height || col >= d_A.width) return;
	int h_reps = d_B.width / d_A.width;
	int v_reps = d_B.height / d_A.height;
	for(int i=0; i < h_reps; i++){
		for(int j=0; j < v_reps; j++){
			d_B.elements[row*d_B.width + col + d_A.width*i + d_B.width*j*d_A.height] = d_A.elements[row*d_A.width + col];
		}
	}
}

void repmat(Matrix A, Matrix B){
// load A and B to device memory
	Matrix d_A;
	Matrix d_B;
	d_A.width = A.width;
	d_B.width = B.width;
	d_A.height = A.height;
	d_B.height = B.height;
	size_t sizeA = A.width * A.height * sizeof(double);
	size_t sizeB = B.width * B.height * sizeof(double);

	cudaError_t err = cudaMalloc(&d_A.elements, sizeA);
	printf("CUDA malloc A: %s\n", cudaGetErrorString(err));
	cudaMemcpy(d_A.elements, A.elements, sizeA, cudaMemcpyHostToDevice);
	printf("Copy A to device: %s\n", cudaGetErrorString(err));

	cudaError_t errB = cudaMalloc(&d_B.elements, sizeB);
	printf("CUDA malloc B: %s\n", cudaGetErrorString(errB));
	cudaMemcpy(d_B.elements, B.elements, sizeB, cudaMemcpyHostToDevice);
	printf("Copy B to device: %s\n", cudaGetErrorString(errB));

// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (A.width + dimBlock.x - 1)/dimBlock.x, (A.height + dimBlock.y - 1)/dimBlock.y );
	repmatKernel<<<dimGrid, dimBlock>>>(d_A, d_B);
	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

// read A from device memory
	err = cudaMemcpy(A.elements, d_A.elements, sizeA, cudaMemcpyDeviceToHost);
	printf("Copy A off of device: %s\n",cudaGetErrorString(err));
	err = cudaMemcpy(B.elements, d_B.elements, sizeB, cudaMemcpyDeviceToHost);
	printf("Copy B off of device: %s\n", cudaGetErrorString(errB));

// free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
}

__global__
void matSubKernel(Matrix d_A, Matrix d_B, Matrix d_C){
int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row > d_A.height || col > d_A.width) return;
	d_C.elements[row*A.width + col] = d_A.elements[row*A.width + col] - d_B.elements[row*A.width + col];
}

void matSub(Matrix A, Matrix B, Matrix C){

// load A, B, and C to device memory
	Matrix d_A;
	Matrix d_B;
	Matrix d_C;
	d_A.width = A.width;
	d_B.width = B.width;
	d_C.width = C.width;
	d_A.height = A.height;
	d_B.height = B.height;
	d_C.height = C.height;
	size_t size = A.width * A.height * sizeof(double);

	cudaError_t err = cudaMalloc(&d_A.elements, size);
	printf("CUDA malloc A: %s\n", cudaGetErrorString(err));
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
	printf("Copy A to device: %s\n", cudaGetErrorString(err));

	cudaError_t err = cudaMalloc(&d_B.elements, size);
	printf("CUDA malloc B: %s\n", cudaGetErrorString(err));
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
	printf("Copy B to device: %s\n", cudaGetErrorString(err));

	cudaError_t err = cudaMalloc(&d_C.elements, size);
	printf("CUDA malloc C: %s\n", cudaGetErrorString(err));
	cudaMemcpy(d_C.elements, C.elements, size, cudaMemcpyHostToDevice);
	printf("Copy C to device: %s\n", cudaGetErrorString(err));

// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (A.width + dimBlock.x - 1)/dimBlock.x, (A.height + dimBlock.y - 1)/dimBlock.y );
	matSubKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

// read A from device memory
	err = cudaMemcpy(A.elements, d_A.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy A off of device: %s\n",cudaGetErrorString(err));
	err = cudaMemcpy(B.elements, d_B.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy B off of device: %s\n", cudaGetErrorString(err));
	err = cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy C off of device: %s\n", cudaGetErrorString(err));

// free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}

__global__
void matAddKernel(Matrix d_A, Matrix d_B, Matrix d_C){

	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= d_A.height || col >= d_A.width) return;
	d_C.elements[row*d_C.width + col] = d_A.elements[row*d_A.width + col] + d_B.elements[row*d_B.width + col];
}

void matAdd(Matrix A, Matrix B, Matrix C){

// load A, B, and C to device memory
	Matrix d_A;
	Matrix d_B;
	Matrix d_C;
	d_A.width = A.width;
	d_B.width = B.width;
	d_C.width = C.width;
	d_A.height = A.height;
	d_B.height = B.height;
	d_C.height = C.height;
	size_t size = A.width * A.height * sizeof(double);

	cudaError_t errA = cudaMalloc(&d_A.elements, size);
	printf("CUDA malloc A: %s\n", cudaGetErrorString(errA));
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
	printf("Copy A to device: %s\n", cudaGetErrorString(errA));

	cudaError_t errB = cudaMalloc(&d_B.elements, size);
	printf("CUDA malloc B: %s\n", cudaGetErrorString(errB));
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
	printf("Copy B to device: %s\n", cudaGetErrorString(errB));

	cudaError_t errC = cudaMalloc(&d_C.elements, size);
	printf("CUDA malloc C: %s\n", cudaGetErrorString(errC));
	cudaMemcpy(d_C.elements, C.elements, size, cudaMemcpyHostToDevice);
	printf("Copy C to device: %s\n", cudaGetErrorString(errC));

// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (A.width + dimBlock.x - 1)/dimBlock.x, (A.height + dimBlock.y - 1)/dimBlock.y );
	matAddKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
	cudaError_t err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

// read A from device memory
	err = cudaMemcpy(A.elements, d_A.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy A off of device: %s\n",cudaGetErrorString(errA));
	err = cudaMemcpy(B.elements, d_B.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy B off of device: %s\n", cudaGetErrorString(errB));
	err = cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy C off of device: %s\n", cudaGetErrorString(errC));

// free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}


__global__
void matTimesScalerKernel(Matrix d_A, Matrix d_B, double C){
int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row > d_A.height || col > d_A.width) return;
	d_B.elements[row*d_A.width + col] = d_A.elements[row*d_A.width + col] * C;
}

void matTimesScaler(Matrix A, Matrix B, double C){

// load A, B, and C to device memory
	Matrix d_A;
	Matrix d_B;
	
	d_A.width = A.width;
	d_B.width = B.width;
	d_A.height = A.height;
	d_B.height = B.height;
	size_t size = A.width * A.height * sizeof(double);

	cudaError_t errA = cudaMalloc(&d_A.elements, size);
	printf("CUDA malloc A: %s\n", cudaGetErrorString(errA));
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
	printf("Copy A to device: %s\n", cudaGetErrorString(errA));

	cudaError_t errB = cudaMalloc(&d_B.elements, size);
	printf("CUDA malloc B: %s\n", cudaGetErrorString(errB));
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
	printf("Copy B to device: %s\n", cudaGetErrorString(errB));

// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (A.width + dimBlock.x - 1)/dimBlock.x, (A.height + dimBlock.y - 1)/dimBlock.y );
	matTimesScalerKernel<<<dimGrid, dimBlock>>>(d_A, d_B, C);
	cudaError_t err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

// read A from device memory
	errA = cudaMemcpy(A.elements, d_A.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy A off of device: %s\n",cudaGetErrorString(errA));
// read B from device memory
	errB = cudaMemcpy(B.elements, d_B.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy B off of device: %s\n", cudaGetErrorString(errB));

// free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
}

__global__
void matPlusScalerKernel(Matrix d_A, Matrix d_B, double C){
int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row > d_A.height || col > d_A.width) return;
	d_B.elements[row*d_A.width + col] = d_A.elements[row*d_A.width + col] + C;
}

void matPlusScaler(Matrix A, Matrix B, double C){

// load A, B, and C to device memory
	Matrix d_A;
	Matrix d_B;
	
	d_A.width = A.width;
	d_B.width = B.width;
	d_A.height = A.height;
	d_B.height = B.height;
	size_t size = A.width * A.height * sizeof(double);

	cudaError_t errA = cudaMalloc(&d_A.elements, size);
	printf("CUDA malloc A: %s\n", cudaGetErrorString(errA));
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
	printf("Copy A to device: %s\n", cudaGetErrorString(errA));

	cudaError_t errB = cudaMalloc(&d_B.elements, size);
	printf("CUDA malloc B: %s\n", cudaGetErrorString(errB));
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
	printf("Copy B to device: %s\n", cudaGetErrorString(errB));

// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (A.width + dimBlock.x - 1)/dimBlock.x, (A.height + dimBlock.y - 1)/dimBlock.y );
	matPlusScalerKernel<<<dimGrid, dimBlock>>>(d_A, d_B, C);
	cudaError_t err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

// read A from device memory
	errA = cudaMemcpy(A.elements, d_A.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy A off of device: %s\n",cudaGetErrorString(errA));
// read B from device memory
	errB = cudaMemcpy(B.elements, d_B.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy B off of device: %s\n", cudaGetErrorString(errB));

// free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
}

__global__
void matDivKernel(Matrix d_A, Matrix d_B, Matrix d_C){
int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row > d_A.height || col > d_A.width) return;
	d_C.elements[row*d_C.width + col] = d_A.elements[row*d_A.width + col] / d_B.elements[row*d_B.width + col];
}

void matDiv(Matrix A, Matrix B, Matrix C){

// load A, B, and C to device memory
	Matrix d_A;
	Matrix d_B;
	Matrix d_C;
	d_A.width = A.width;
	d_B.width = B.width;
	d_C.width = C.width;
	d_A.height = A.height;
	d_B.height = B.height;
	d_C.height = C.height;
	size_t size = A.width * A.height * sizeof(double);

	cudaError_t errA = cudaMalloc(&d_A.elements, size);
	printf("CUDA malloc A: %s\n", cudaGetErrorString(errA));
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
	printf("Copy A to device: %s\n", cudaGetErrorString(errA));

	cudaError_t errB = cudaMalloc(&d_B.elements, size);
	printf("CUDA malloc B: %s\n", cudaGetErrorString(errB));
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
	printf("Copy B to device: %s\n", cudaGetErrorString(errB));

	cudaError_t errC = cudaMalloc(&d_C.elements, size);
	printf("CUDA malloc C: %s\n", cudaGetErrorString(errC));
	cudaMemcpy(d_C.elements, C.elements, size, cudaMemcpyHostToDevice);
	printf("Copy C to device: %s\n", cudaGetErrorString(errC));

// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (A.width + dimBlock.x - 1)/dimBlock.x, (A.height + dimBlock.y - 1)/dimBlock.y );
	matDivKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
	cudaError_t err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

// read A from device memory
	errA = cudaMemcpy(A.elements, d_A.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy A off of device: %s\n",cudaGetErrorString(errA));
// read B from device memory
	errB = cudaMemcpy(B.elements, d_B.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy B off of device: %s\n", cudaGetErrorString(errB));
// read C from device memory
	errC = cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy C off of device: %s\n", cudaGetErrorString(errC));

// free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}

// matrix zeros kernel called by getCol()
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
	size_t size = In.width * In.height * sizeof(double);
	cudaError_t err = cudaMalloc(&d_In.elements, size);
	printf("CUDA malloc In: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_In.elements, In.elements, size, cudaMemcpyHostToDevice);	
	printf("Copy input matrix to device: %s\n", cudaGetErrorString(err));
	
	// allocate Out in device memory
	Matrix d_Out;
  d_Out.width = Out.width; d_Out.height = Out.height;
  size = Out.width * Out.height * sizeof(double);
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

// matrix zeros kernel called by getRow()
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
	size_t size = In.width * In.height * sizeof(double);
	cudaError_t err = cudaMalloc(&d_In.elements, size);
	printf("CUDA malloc In: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_In.elements, In.elements, size, cudaMemcpyHostToDevice);	
	printf("Copy input matrix to device: %s\n", cudaGetErrorString(err));
	
	// allocate Out in device memory
	Matrix d_Out;
  d_Out.width = Out.width; d_Out.height = Out.height;
  size = Out.width * Out.height * sizeof(double);
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

// matrix zeros kernel called by indexOfElement()
__global__
void indexOfElementKernel(Matrix d_A, double element, int *index) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row > d_A.height || col > d_A.width) return;
	int idx = row*d_A.width+col;
	if (d_A.elements[idx] == element)
		*(index) = idx;
}

int indexOfElement(Matrix A, double element) {
	int index;	
	// load A to device memory
	Matrix d_A;
	d_A.width = A.width;
	d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(double);
	cudaError_t err = cudaMalloc(&d_A.elements, size);
	printf("CUDA malloc A: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);	
	printf("Copy A to device: %s\n", cudaGetErrorString(err));

	// load index to device memory
	int *d_index;
	cudaMemset(d_index, -1, sizeof(int));
	err = cudaMalloc(&d_index, sizeof(int));
	printf("CUDA malloc index; %s\n", cudaGetErrorString(err));
	cudaMemcpy(d_index, &index, sizeof(int), cudaMemcpyHostToDevice);
	printf("Copy index to device: %s\n", cudaGetErrorString(err));
	
	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (A.width + dimBlock.x - 1)/dimBlock.x, (A.height + dimBlock.y - 1)/dimBlock.y );
	indexOfElementKernel<<<dimGrid, dimBlock>>>(d_A, element, d_index);
	err = cudaThreadSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read index from device memory
	err = cudaMemcpy(&index, d_index, sizeof(int), cudaMemcpyDeviceToHost);
	printf("Copy index off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_A.elements);
	cudaFree(d_index);
	return index;
}

__global__
void maxReduceKernel(double *elements, int size, double *d_part) {
	// Reduction max, works for any blockDim.x:
	int  thread2;
	double temp;
	__shared__ double sdata[BLOCK_SIZE];
	
	// Load max from global memory
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		sdata[threadIdx.x] = elements[idx];
	else
		sdata[threadIdx.x] = DBL_MIN;
	
	// Synchronize to make sure data is loaded before starting the comparison
  __syncthreads();

	int nTotalThreads = BLOCK_SIZE;	// Total number of threads, rounded up to the next power of two
	 
	while(nTotalThreads > 1) {
		int halfPoint = (nTotalThreads >> 1);	// divide by two
		// only the first half of the threads will be active.
	 
		if (threadIdx.x < halfPoint) {
			thread2 = threadIdx.x + halfPoint;

			// Skipping the fictious threads blockDim.x ... blockDim_2-1
			if (thread2 < blockDim.x) {
				// Get the shared value stored by another thread 
				temp = sdata[thread2];
				if (temp > sdata[threadIdx.x]) 
					 sdata[threadIdx.x] = temp;
			}
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

double maxOfMatrix(Matrix A) {
	cudaEvent_t start, stop;
	float time;
	// create events and start the timer
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0 );

	// load A to device memory
	Matrix d_A;
	d_A.width = A.width;
	d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(double);
	cudaError_t err = cudaMalloc(&d_A.elements, size);
	printf("CUDA malloc A: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);	
	printf("Copy A to device: %s\n", cudaGetErrorString(err));

	// load d_part to device memory
	double *d_part;
	err = cudaMalloc(&d_part, BLOCK_SIZE*sizeof(double));
	printf("CUDA malloc d_part; %s\n", cudaGetErrorString(err));
	err = cudaMemset(d_part, DBL_MIN, BLOCK_SIZE*sizeof(double));
	printf("CUDA memset d_part to DBL_MIN: %s\n", cudaGetErrorString(err));

	// load d_max to device memory
	double *d_max;
	err = cudaMalloc(&d_max, sizeof(double));
	printf("CUDA malloc d_max; %s\n", cudaGetErrorString(err));
	err = cudaMemset(d_max, DBL_MIN, sizeof(double));
	printf("CUDA memset d_max to DBL_MIN: %s\n", cudaGetErrorString(err));

	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid((A.width*A.height + dimBlock.x - 1)/dimBlock.x);
	//int blockDim_2 = NearestPowerOf2(d_A.width*d_A.height);
	//printf("nearest power of 2 (blockDim_2): %d\n",blockDim_2);
	// first pass
	maxReduceKernel<<<dimGrid, dimBlock>>>(d_A.elements, d_A.width*d_A.height, d_part);
	err = cudaThreadSynchronize();
	printf("Run kernel 1st pass: %s\n", cudaGetErrorString(err));
	// second pass
	dimGrid = dim3(1);
	maxReduceKernel<<<dimGrid, dimBlock>>>(d_part, BLOCK_SIZE, d_max);
	err = cudaThreadSynchronize();
	printf("Run kernel 2nd pass: %s\n", cudaGetErrorString(err));

	// read max from device memory
	double max;
	err = cudaMemcpy(&max, d_max, sizeof(double), cudaMemcpyDeviceToHost);
	printf("Copy max off of device: %s\n",cudaGetErrorString(err));
	
	// stop the timer
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );

	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	printf("Time elapsed: %f ms\n", time);



	// free device memory
	cudaFree(d_A.elements);
	cudaFree(d_max);
	return max;
}

// matrix zeros kernel called by zeros()
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
	size_t size = In.width * In.height * sizeof(double);
	cudaError_t err = cudaMalloc(&d_In.elements, size);
	printf("CUDA malloc In: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_In.elements, In.elements, size, cudaMemcpyHostToDevice);	
	printf("Copy input matrix to device: %s\n", cudaGetErrorString(err));
	
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
	printf("Run kernel: %s\n", cudaGetErrorString(err));

	// read Out from device memory
	err = cudaMemcpy(Out.elements, d_Out.elements, size, cudaMemcpyDeviceToHost);
	printf("Copy output matrix off of device: %s\n",cudaGetErrorString(err));

	// free device memory
	cudaFree(d_In.elements);
	cudaFree(d_Out.elements);
}

__global__
void maxReduceKernel(double *elements, int size, double *d_part) {
	// Reduction max, works for any blockDim.x:
	int  thread2;
	double temp;
	__shared__ double sdata[BLOCK_SIZE];
	
	// Load max from global memory
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		sdata[threadIdx.x] = elements[idx];
	else
		sdata[threadIdx.x] = DBL_MIN;
	
	// Synchronize to make sure data is loaded before starting the comparison
  __syncthreads();

	int nTotalThreads = BLOCK_SIZE;	// Total number of threads, rounded up to the next power of two
	 
	while(nTotalThreads > 1) {
		int halfPoint = (nTotalThreads >> 1);	// divide by two
		// only the first half of the threads will be active.
	 
		if (threadIdx.x < halfPoint) {
			thread2 = threadIdx.x + halfPoint;

			// Skipping the fictious threads blockDim.x ... blockDim_2-1
			if (thread2 < blockDim.x) {
				// Get the shared value stored by another thread 
				temp = sdata[thread2];
				if (temp > sdata[threadIdx.x]) 
					 sdata[threadIdx.x] = temp;
			}
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

double maxOfArray(double* A, int elements) {
	cudaEvent_t start, stop;
	float time;
	// create events and start the timer
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0 );

	// load A to device memory
	double* d_A;
	size_t size = elements * sizeof(double);
	cudaError_t err = cudaMalloc(&d_A, size);
	printf("CUDA malloc A: %s\n", cudaGetErrorString(err));	
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);	
	printf("Copy A to device: %s\n", cudaGetErrorString(err));

	// load d_part to device memory
	double *d_part;
	err = cudaMalloc(&d_part, BLOCK_SIZE*sizeof(double));
	printf("CUDA malloc d_part; %s\n", cudaGetErrorString(err));
	err = cudaMemset(d_part, DBL_MIN, BLOCK_SIZE*sizeof(double));
	printf("CUDA memset d_part to DBL_MIN: %s\n", cudaGetErrorString(err));

	// load d_max to device memory
	double *d_max;
	err = cudaMalloc(&d_max, sizeof(double));
	printf("CUDA malloc d_max; %s\n", cudaGetErrorString(err));
	err = cudaMemset(d_max, DBL_MIN, sizeof(double));
	printf("CUDA memset d_max to DBL_MIN: %s\n", cudaGetErrorString(err));

	// invoke kernel
	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid((elements + dimBlock.x - 1)/dimBlock.x);
	//int blockDim_2 = NearestPowerOf2(d_A.width*d_A.height);
	//printf("nearest power of 2 (blockDim_2): %d\n",blockDim_2);
	// first pass
	maxReduceKernel<<<dimGrid, dimBlock>>>(d_A, elements, d_part);
	err = cudaThreadSynchronize();
	printf("Run kernel 1st pass: %s\n", cudaGetErrorString(err));
	// second pass
	dimGrid = dim3(1);
	maxReduceKernel<<<dimGrid, dimBlock>>>(d_part, BLOCK_SIZE, d_max);
	err = cudaThreadSynchronize();
	printf("Run kernel 2nd pass: %s\n", cudaGetErrorString(err));

	// read max from device memory
	double max;
	err = cudaMemcpy(&max, d_max, sizeof(double), cudaMemcpyDeviceToHost);
	printf("Copy max off of device: %s\n",cudaGetErrorString(err));
	
	// stop the timer
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );

	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	printf("Time elapsed: %f ms\n", time);

	// free device memory
	cudaFree(d_A);
	cudaFree(d_max);
	return max;
}
