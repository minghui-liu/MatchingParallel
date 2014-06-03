#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 32 

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
  int width;
  int height;
	double* elements;
} Matrix;

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

//usage : zeros height width
int main(int argc, char* argv[]) {
	Matrix A;
	int a1, a2;
	// Read some values from the commandline
	a1 = atoi(argv[1]); /* Height of A */
	a2 = atoi(argv[2]); /* Width of A */
	A.height = a1;
	A.width = a2;
	A.elements = (double*)malloc(A.width * A.height * sizeof(double));
	// give A random values
	for(int i = 0; i < A.height; i++)
		for(int j = 0; j < A.width; j++)
			A.elements[i*A.width + j] = ((double)rand()/(double)(RAND_MAX)) * 10;
	printMatrix(A);
	// call zeros
	zeros(A);
	printMatrix(A);
}
