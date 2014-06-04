#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK_SIZE 16

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
	int width;
	int height;
	double* elements;
} Matrix;

// check if a square matrix is symmetric
__global__
void isSymmetricKernel(Matrix d_A, Matrix d_B){

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= d_A.height || col >= d_A.width) return;
	if(d_A.elements[row*d_A.width + col] == d_A.elements[row + col*d_A.width])
		return;
	else
		d_B.elements[row*d_B.width + col] = 1;
}

bool isSymmetric(Matrix A, Matrix B) {
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

	for(int i=1; i<A.height; i++){
		for( int j=0; j<i; j++){
			if(B.elements[i*B.width + j] != 0){
				return false;
			}
		}
	}
	return true;
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

//usage : isSymmetric height width
int main(int argc, char* argv[]) {

	srand(time(0));	

	Matrix A;
	Matrix B;
	int a1, a2;
	// Read some values from the commandline
	a1 = atoi(argv[1]); /* Height of A */
	a2 = atoi(argv[2]); /* Width of A */
	A.height = a1;
	B.height = a1;
	A.width = a2;
	B.width = a2;	
	A.elements = (double*)malloc(A.width * A.height * sizeof(double));
	B.elements = (double*)malloc(B.width * B.height * sizeof(double));
	// give A random values and B zeros
	for(int i = 0; i < A.height; i++){
		for(int j = 0; j <= i; j++){
			A.elements[i*A.width + j] = ((double)rand()/(double)(RAND_MAX)) * 10;
			A.elements[j*A.width + i] = A.elements[i*A.width + j];
			B.elements[i*B.width + j] = 0;
		}
	}
	// call isSymmetric
	bool test =	isSymmetric(A, B);
	printMatrix(A);
	printMatrix(B);
	printf("%i \n", test);
}
