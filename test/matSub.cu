#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 16

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
	int width;
	int height;
	double* elements;
} Matrix;

__global__
void matSubKernel(Matrix d_A, Matrix d_B, Matrix d_C){

	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= d_A.height || col >= d_A.width) return;
	d_C.elements[row*d_C.width + col] = d_A.elements[row*d_A.width + col] - d_B.elements[row*d_B.width + col];
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
	matSubKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
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

//usage : matSub height width
int main(int argc, char* argv[]) {

	srand(time(0));	

	Matrix A;
	Matrix B;
	Matrix C;
	int a1, a2;
	// Read some values from the commandline
	a1 = atoi(argv[1]); /* Height of A */
	a2 = atoi(argv[2]); /* Width of A */
	A.height = a1;
	B.height = a1;
	C.height = a1;
	A.width = a2;
	B.width = a2;
	C.width = a2;
	A.elements = (double*)malloc(A.width * A.height * sizeof(double));
	B.elements = (double*)malloc(B.width * B.height * sizeof(double));
	C.elements = (double*)malloc(C.width * C.height * sizeof(double));
	// give A random values
	for(int i = 0; i < A.height; i++)
		for(int j = 0; j < A.width; j++)
			A.elements[i*A.width + j] = ((double)rand()/(double)(RAND_MAX)) * 10;
	// give B random values
	for(int i = 0; i < B.height; i++)
		for(int j = 0; j < B.width; j++)
			B.elements[i*B.width + j] = ((double)rand()/(double)(RAND_MAX)) * 10;
	// call matSub
	matSub(A, B, C);
	printMatrix(A);
	printMatrix(B);
	printMatrix(C);
}
