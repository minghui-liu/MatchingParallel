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

//usage : repmat height width
int main(int argc, char* argv[]) {
	Matrix A;
	Matrix B;
	int a1, a2, b1, b2;
	// Read some values from the commandline
	a1 = atoi(argv[1]); /* Height of A */
	a2 = atoi(argv[2]); /* Width of A */
	b1 = atoi(argv[3]); /* Height of B */
	b2 = atoi(argv[4]); /* Width of B */
	A.height = a1;
	B.height = b1;
	A.width = a2;
	B.width = b2;
	A.elements = (double*)malloc(A.width * A.height * sizeof(double));
	B.elements = (double*)malloc(B.width * B.height * sizeof(double));
	// give A random values
	for(int i = 0; i < A.height; i++)
		for(int j = 0; j < A.width; j++)
			A.elements[i*A.width + j] = ((double)rand()/(double)(RAND_MAX)) * 10;
	// call repmat
	repmat(A, B);
	printMatrix(A);
	printMatrix(B);
}
